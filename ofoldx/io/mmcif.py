r"""Standard mmCIF projection of the canonical tensor molecular graph."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch

from ..data._cif import format_chemical_stereo
from ..data._component_signatures import (
    _graph_component_signatures,
    _realized_component_signature,
)
from ..data.components import (
    get_builtin_component_polymer_type,
    get_component,
)
from ..data.elements import ATOMIC_NUMBER_BY_ELEMENT
from ..data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MOLECULE_BRANCHED,
    MOLECULE_MACROLIDE,
    MOLECULE_NON_POLYMER,
    MOLECULE_WATER,
    PolymerType,
)
from ..data.structure import (
    RESOLUTION_KIND_EM_RECONSTRUCTION,
    RESOLUTION_KIND_REFINEMENT,
    RESOLUTION_KIND_REFLECTIONS,
    Structure,
)
from .records import (
    unique_identifier,
)


def write_mmcif(structure: Structure, path: Path) -> None:
    r"""Write the observed projection and molecular declarations as standard mmCIF."""
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    from ..data._cif_chemistry import (
        _cif_bond_order,
        _component_atom_groups,
        _component_leaving_atom,
        _remove_component_atoms,
    )

    system = structure.system.cpu()  # type: ignore[attr-defined]
    coordinates = structure.coordinates.detach().cpu()
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    occupancy = structure.occupancy.detach().cpu()
    occupancy_mask = structure.occupancy_mask.detach().cpu()
    b_factor = structure.b_factor.detach().cpu()
    b_factor_mask = structure.b_factor_mask.detach().cpu()
    residue_id = structure.residue_id.detach().cpu()
    insertion_code = structure.insertion_code.detach().cpu()
    source_formal_charge_mask = structure.source_formal_charge_mask.detach().cpu()
    resolution = structure.resolution.detach().cpu()
    resolution_kind = structure.resolution_kind.detach().cpu()
    instance_entity = system.instance_entity.detach()
    entity_type = system.entity_type.detach()
    entity_molecule_type = system.entity_molecule_type.detach()
    instance_component_ptr = system.instance_component_ptr.detach()
    component_symbol = system.component_symbol.detach()
    component_topology_complete = system.component_topology_complete.detach()
    component_atom_ptr = system.component_atom_ptr.detach()
    atom_symbol = system.atom_symbol.detach()
    atom_stereo = system.atom_stereo.detach()
    atomic_number = system.atomic_number.detach()
    isotope = system.isotope.detach()
    isotope_known = system.isotope_known.detach()
    formal_charge = system.formal_charge.detach()
    formal_charge_known = system.formal_charge_known.detach()
    atom_aromatic = system.atom_aromatic.detach()
    atom_aromatic_known = system.atom_aromatic_known.detach()
    reference_coordinates = system.reference_coordinates.detach()
    reference_coordinate_mask = system.reference_coordinate_mask.detach()
    edge_index = system.edge_index.detach()
    bond_order = system.bond_order.detach()
    bond_order_known = system.bond_order_known.detach()
    bond_aromatic = system.bond_aromatic.detach()
    bond_aromatic_known = system.bond_aromatic_known.detach()
    bond_kind = system.bond_kind.detach()
    bond_direction = system.bond_direction.detach()
    bond_role = system.bond_role.detach()
    bond_stereo = system.bond_stereo.detach()
    bond_stereo_index = system.bond_stereo_index.detach()
    element_by_number = {number: element for element, number in ATOMIC_NUMBER_BY_ELEMENT.items()}

    instance_count = int(system.instance_symbol.shape[0])
    component_count = int(component_symbol.shape[0])
    atom_count = int(atom_symbol.shape[0])
    component_instance = [0] * component_count
    atom_component = [0] * atom_count
    for instance_index in range(instance_count):
        start = int(instance_component_ptr[instance_index])
        stop = int(instance_component_ptr[instance_index + 1])
        for component_index in range(start, stop):
            component_instance[component_index] = instance_index
            atom_start = int(component_atom_ptr[component_index])
            atom_stop = int(component_atom_ptr[component_index + 1])
            atom_component[atom_start:atom_stop] = [component_index] * (atom_stop - atom_start)

    label_asym_by_instance: dict[int, str] = {}
    used_asym_ids: set[str] = set()
    for instance_index in range(instance_count):
        label_asym_by_instance[instance_index] = unique_identifier(
            system.instance_id(instance_index),
            used_asym_ids,
        )
    entity_id_by_index: dict[int, str] = {}
    used_entity_ids: set[str] = set()
    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id_by_index[entity_index] = unique_identifier(
            system.entity_id(entity_index),
            used_entity_ids,
        )

    def component_id(component_index: int) -> str:
        return system.symbols[int(component_symbol[component_index])]

    def atom_id(atom_index: int) -> str:
        return system.symbols[int(atom_symbol[atom_index])]

    def element(atom_index: int) -> str:
        number = int(atomic_number[atom_index])
        if number == 1 and bool(isotope_known[atom_index]):
            if int(isotope[atom_index]) == 2:
                return "D"
            if int(isotope[atom_index]) == 3:
                return "T"
        return element_by_number[number]

    def source_value(values: torch.Tensor, index: int) -> str | None:
        symbol_index = int(values[index])
        return None if symbol_index < 0 else structure.source_symbols[symbol_index]

    def order_name(index: int) -> str:
        if not bool(bond_order_known[index]):
            return "?"
        if bool(bond_aromatic_known[index]) and bool(bond_aromatic[index]) and float(bond_order[index]) == 1.5:
            return "arom"
        try:
            return {
                1.0: "sing",
                2.0: "doub",
                3.0: "trip",
                4.0: "quad",
            }[float(bond_order[index])]
        except KeyError:
            raise ValueError(f"mmCIF cannot encode known bond order {float(bond_order[index])!r}") from None

    def connection_order_name(index: int) -> str:
        value = order_name(index)
        decoded_order, decoded_aromatic = _cif_bond_order(None if value == "?" else value)
        # Mirror the reader: in `struct_conn` an explicit order also states non-aromaticity.
        if decoded_order is not None and decoded_aromatic is None:
            decoded_aromatic = False
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError(
                "mmCIF struct_conn cannot encode partially known or inconsistent bond order and aromaticity"
            )
        return value

    def component_bond_values(index: int) -> tuple[str, str]:
        order_value = order_name(index)
        expected_order = float(bond_order[index]) if bool(bond_order_known[index]) else None
        expected_aromatic = bool(bond_aromatic[index]) if bool(bond_aromatic_known[index]) else None
        decoded_order, order_aromatic = _cif_bond_order(None if order_value == "?" else order_value)
        if expected_aromatic is None:
            decoded_aromatic = order_aromatic
            aromatic_value = "?"
        else:
            if order_aromatic is not None and order_aromatic != expected_aromatic:
                raise ValueError("mmCIF chem_comp_bond cannot encode inconsistent bond order and aromaticity")
            decoded_aromatic = expected_aromatic
            aromatic_value = "Y" if expected_aromatic else "N"
        if (decoded_order, decoded_aromatic) != (
            expected_order,
            expected_aromatic,
        ):
            raise ValueError("mmCIF chem_comp_bond cannot encode partially known bond order and aromaticity")
        return order_value, aromatic_value

    entity_columns: dict[str, list[object]] = {"id": [], "type": []}
    struct_asym_columns: dict[str, list[object]] = {"id": [], "entity_id": []}
    entity_poly_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    entity_poly_seq_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "mon_id": [],
        "hetero": [],
    }
    poly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "seq_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "auth_seq_num": [],
        "pdb_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
        "hetero": [],
    }
    branch_columns: dict[str, list[object]] = {"entity_id": [], "type": []}
    nonpoly_columns: dict[str, list[object]] = {
        "entity_id": [],
        "name": [],
        "comp_id": [],
    }
    nonpoly_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "ndb_seq_num": [],
        "pdb_seq_num": [],
        "auth_seq_num": [],
        "pdb_mon_id": [],
        "auth_mon_id": [],
        "pdb_strand_id": [],
        "pdb_ins_code": [],
    }
    branch_list_columns: dict[str, list[object]] = {
        "entity_id": [],
        "num": [],
        "comp_id": [],
        "hetero": [],
    }
    branch_link_columns: dict[str, list[object]] = {
        "entity_id": [],
        "link_id": [],
        "entity_branch_list_num_1": [],
        "comp_id_1": [],
        "atom_id_1": [],
        "leaving_atom_id_1": [],
        "entity_branch_list_num_2": [],
        "comp_id_2": [],
        "atom_id_2": [],
        "leaving_atom_id_2": [],
        "value_order": [],
    }
    branch_scheme_columns: dict[str, list[object]] = {
        "asym_id": [],
        "entity_id": [],
        "mon_id": [],
        "num": [],
        "pdb_asym_id": [],
        "pdb_mon_id": [],
        "pdb_seq_num": [],
        "pdb_ins_code": [],
        "auth_asym_id": [],
        "auth_mon_id": [],
        "auth_seq_num": [],
    }

    entity_type_name = {
        ENTITY_PROTEIN: "polypeptide(L)",
        ENTITY_RNA: "polyribonucleotide",
        ENTITY_DNA: "polydeoxyribonucleotide",
        ENTITY_DNA_RNA_HYBRID: "polydeoxyribonucleotide/polyribonucleotide hybrid",
        ENTITY_PNA: "peptide nucleic acid",
    }
    first_instance_by_entity: dict[int, int] = {}
    instances_by_entity: dict[int, list[int]] = {}
    branch_leaving_by_endpoint: dict[tuple[int, int], str] = {}
    branch_leaving_by_component: dict[int, frozenset[str]] = {}
    assembly_leaving_by_endpoint: dict[tuple[int, int], tuple[str, frozenset[str]]] = {}
    assembly_leaving_by_component: dict[int, frozenset[str]] = {}
    for instance_index in range(instance_count):
        entity_index = int(instance_entity[instance_index])
        first_instance_by_entity.setdefault(entity_index, instance_index)
        instances_by_entity.setdefault(entity_index, []).append(instance_index)
        struct_asym_columns["id"].append(label_asym_by_instance[instance_index])
        struct_asym_columns["entity_id"].append(entity_id_by_index[entity_index])

    def instance_components(instance_index: int) -> tuple[int, ...]:
        return tuple(
            range(
                int(instance_component_ptr[instance_index]),
                int(instance_component_ptr[instance_index + 1]),
            )
        )

    polymer_pairs = {
        tuple(
            sorted(
                (
                    atom_component[int(edge_index[0, edge])],
                    atom_component[int(edge_index[1, edge])],
                )
            )
        )
        for edge in range(edge_index.shape[1])
        if int(bond_role[edge]) == BOND_ROLE_POLYMER
    }

    def local_atom(instance_index: int, atom_index: int) -> tuple[int, str]:
        component_index = atom_component[atom_index]
        if component_instance[component_index] != instance_index:
            raise ValueError("an entity-owned bond crosses molecular instances")
        return (
            component_index - int(instance_component_ptr[instance_index]),
            atom_id(atom_index),
        )

    def owned_edge_signature(
        instance_index: int,
        role: int,
    ) -> tuple[tuple[object, ...], ...]:
        rows = []
        for edge in range(edge_index.shape[1]):
            if int(bond_role[edge]) != role:
                continue
            atom_a = int(edge_index[0, edge])
            atom_b = int(edge_index[1, edge])
            if component_instance[atom_component[atom_a]] != instance_index:
                continue
            endpoint_a = local_atom(instance_index, atom_a)
            endpoint_b = local_atom(instance_index, atom_b)
            stereo_atoms = tuple(
                (None if int(atom) < 0 else local_atom(instance_index, int(atom)))
                for atom in bond_stereo_index[:, edge]
            )
            direction = int(bond_direction[edge])
            leaving_atoms: tuple[str | None, str | None] | None = None
            if role == BOND_ROLE_ENTITY:
                leaving_atoms = (
                    branch_leaving_by_endpoint.get((edge, 0)),
                    branch_leaving_by_endpoint.get((edge, 1)),
                )
            if endpoint_b < endpoint_a:
                endpoint_a, endpoint_b = endpoint_b, endpoint_a
                stereo_atoms = stereo_atoms[::-1]
                direction = -direction
                if leaving_atoms is not None:
                    leaving_atoms = leaving_atoms[::-1]
            stereo = int(bond_stereo[edge])
            row = (
                endpoint_a,
                endpoint_b,
                (
                    bool(bond_order_known[edge]),
                    float(bond_order[edge]),
                ),
                (
                    bool(bond_aromatic_known[edge]),
                    bool(bond_aromatic[edge]),
                ),
                int(bond_kind[edge]),
                direction,
                None if stereo < 0 else system.symbols[stereo],
                stereo_atoms,
            )
            rows.append(row if leaving_atoms is None else (*row, leaving_atoms))
        return tuple(sorted(rows))

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        atom_a = int(edge_index[0, edge])
        atom_b = int(edge_index[1, edge])
        component_a = atom_component[atom_a]
        component_b = atom_component[atom_b]
        instance_a = component_instance[component_a]
        instance_b = component_instance[component_b]
        if role == BOND_ROLE_COMPONENT and component_a != component_b:
            raise ValueError("a component-owned bond crosses component occurrences")
        if role in {BOND_ROLE_POLYMER, BOND_ROLE_ENTITY} and instance_a != instance_b:
            raise ValueError("an entity-owned bond crosses molecular instances")
        if role == BOND_ROLE_POLYMER and component_a == component_b:
            raise ValueError("a polymer-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY and component_a == component_b:
            raise ValueError("an entity-owned bond must join component occurrences")
        if role == BOND_ROLE_ENTITY:
            entity_index = int(instance_entity[instance_a])
            if (
                int(entity_type[entity_index]) != ENTITY_MOLECULE
                or int(entity_molecule_type[entity_index]) != MOLECULE_BRANCHED
            ):
                raise ValueError("only a branched molecule can carry entity-owned bonds")

    branch_endpoints_by_component: dict[int, list[tuple[int, int, str]]] = {}
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_ENTITY:
            continue
        for side in range(2):
            atom = int(edge_index[side, edge])
            branch_endpoints_by_component.setdefault(
                atom_component[atom],
                [],
            ).append((edge, side, atom_id(atom)))

    for component_index, endpoints in branch_endpoints_by_component.items():
        try:
            component = get_component(component_id(component_index))
        except KeyError:
            continue
        if component.bonds is None:
            raise ValueError(f"component {component.component_id!r} has no known bond graph")
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        present = {atom_id(atom) for atom in range(start, stop)}
        missing = {atom.atom_id for atom in component.atoms} - present
        selected: set[str] = set()
        groups = _component_atom_groups(component, missing)
        for edge, side, endpoint in endpoints:
            candidates = [
                group
                for group, boundary in groups
                if boundary == frozenset({endpoint}) and not group.intersection(selected)
            ]
            if len(candidates) > 1:
                raise ValueError(f"mmCIF cannot infer a unique leaving group for {component.component_id!r}:{endpoint}")
            if candidates:
                group = candidates[0]
                selected.update(group)
                branch_leaving_by_endpoint[(edge, side)] = _component_leaving_atom(component, group, endpoint)
        branch_leaving_by_component[component_index] = frozenset(selected)

    assembly_endpoints_by_component: dict[int, list[tuple[int, int, str]]] = {}
    for edge in range(edge_index.shape[1]):
        if int(bond_role[edge]) != BOND_ROLE_ASSEMBLY:
            continue
        for side in range(2):
            atom = int(edge_index[side, edge])
            assembly_endpoints_by_component.setdefault(atom_component[atom], []).append((edge, side, atom_id(atom)))

    for component_index, endpoints in assembly_endpoints_by_component.items():
        try:
            component = get_component(component_id(component_index))
        except KeyError:
            continue
        instance_index = component_instance[component_index]
        component_start = int(instance_component_ptr[instance_index])
        component_stop = int(instance_component_ptr[instance_index + 1])
        ordinary_leaving = set(branch_leaving_by_component.get(component_index, ()))
        if (
            component_index > component_start
            and (component_index - 1, component_index) in polymer_pairs
            and component.previous_attachment is not None
        ):
            ordinary_leaving.update(component.previous_attachment.leaving_atom_ids)
        if (
            component_index + 1 < component_stop
            and (component_index, component_index + 1) in polymer_pairs
            and component.next_attachment is not None
        ):
            ordinary_leaving.update(component.next_attachment.leaving_atom_ids)
        ordinary_product = _remove_component_atoms(component, ordinary_leaving)
        start = int(component_atom_ptr[component_index])
        stop = int(component_atom_ptr[component_index + 1])
        present = {atom_id(atom) for atom in range(start, stop)}
        missing = {atom.atom_id for atom in ordinary_product.atoms} - present
        groups = _component_atom_groups(ordinary_product, missing)
        assembly_selected: set[str] = set()
        for edge, side, endpoint in endpoints:
            candidates = [
                group
                for group, boundary in groups
                if boundary == frozenset({endpoint}) and not group.intersection(assembly_selected)
            ]
            if len(candidates) > 1:
                raise ValueError(f"mmCIF cannot infer a unique leaving group for {component.component_id!r}:{endpoint}")
            if candidates:
                group = candidates[0]
                assembly_selected.update(group)
                assembly_leaving_by_endpoint[(edge, side)] = (
                    _component_leaving_atom(ordinary_product, group, endpoint),
                    group,
                )
        assembly_leaving_by_component[component_index] = frozenset(assembly_selected)

    for entity_index in range(int(system.entity_symbol.shape[0])):
        entity_id = entity_id_by_index[entity_index]
        type_code = int(entity_type[entity_index])
        first_instance = first_instance_by_entity[entity_index]
        components = list(instance_components(first_instance))
        component_start = components[0] if components else int(instance_component_ptr[first_instance])
        component_ids_by_position: list[tuple[str, ...]] = []
        for instance_index in instances_by_entity[entity_index]:
            if len(instance_components(instance_index)) != len(components):
                raise ValueError(f"mmCIF entity {entity_id!r} has inconsistent component-axis lengths")
        for local_index in range(len(components)):
            component_ids = tuple(
                dict.fromkeys(
                    component_id(instance_components(instance_index)[local_index])
                    for instance_index in instances_by_entity[entity_index]
                )
            )
            component_ids_by_position.append(component_ids)
        if type_code == ENTITY_MOLECULE:
            if any(len(component_ids) != 1 for component_ids in component_ids_by_position):
                raise ValueError(f"mmCIF cannot encode divergent molecule instances for entity {entity_id!r}")
            first_entity_edges = owned_edge_signature(
                first_instance,
                BOND_ROLE_ENTITY,
            )
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY) != first_entity_edges
                for instance_index in instances_by_entity[entity_index][1:]
            ):
                raise ValueError(f"mmCIF cannot encode divergent entity bonds for entity {entity_id!r}")
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_POLYMER)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a molecule cannot carry polymer-owned bonds")
            molecule_type = int(entity_molecule_type[entity_index])
            if molecule_type == MOLECULE_BRANCHED:
                entity_category = "branched"
            elif molecule_type in {
                MOLECULE_MACROLIDE,
                MOLECULE_NON_POLYMER,
                MOLECULE_WATER,
            }:
                if molecule_type == MOLECULE_MACROLIDE:
                    entity_category = "macrolide"
                elif molecule_type == MOLECULE_WATER:
                    entity_category = "water"
                else:
                    entity_category = "non-polymer"
                if len(components) != 1 or first_entity_edges:
                    raise ValueError(
                        f"mmCIF {entity_category} entities require one component and no entity-owned bonds"
                    )
            else:
                raise ValueError(f"unsupported molecule type code {molecule_type}")
        else:
            if any(
                owned_edge_signature(instance_index, BOND_ROLE_ENTITY)
                for instance_index in instances_by_entity[entity_index]
            ):
                raise ValueError("a polymer cannot carry entity-owned bonds")
            for instance_index in instances_by_entity[entity_index]:
                expected_polymer_edges = []
                occurrence_components = instance_components(instance_index)
                for local_index, (left_index, right_index) in enumerate(
                    zip(occurrence_components, occurrence_components[1:])
                ):
                    try:
                        left = get_component(component_id(left_index))
                        right = get_component(component_id(right_index))
                    except KeyError:
                        continue
                    left_start = int(component_atom_ptr[left_index])
                    left_stop = int(component_atom_ptr[left_index + 1])
                    right_start = int(component_atom_ptr[right_index])
                    right_stop = int(component_atom_ptr[right_index + 1])
                    left_attachment = left.next_attachment
                    right_attachment = right.previous_attachment
                    if left_attachment is None or right_attachment is None:
                        continue
                    observed_left = {
                        atom_id(atom_index)
                        for atom_index in range(left_start, left_stop)
                        if bool(coordinate_mask[atom_index])
                    }
                    observed_right = {
                        atom_id(atom_index)
                        for atom_index in range(right_start, right_stop)
                        if bool(coordinate_mask[atom_index])
                    }
                    if (
                        left_attachment.leaving_atom_ids & observed_left
                        or right_attachment.leaving_atom_ids & observed_right
                    ):
                        continue
                    endpoint_a = (local_index, left_attachment.atom_id)
                    endpoint_b = (
                        local_index + 1,
                        right_attachment.atom_id,
                    )
                    expected_polymer_edges.append(
                        (
                            endpoint_a,
                            endpoint_b,
                            (True, 1.0),
                            (True, False),
                            BOND_KIND_COVALENT,
                            BOND_DIRECTION_UNDIRECTED,
                            None,
                            (None, None),
                        )
                    )
                if owned_edge_signature(
                    instance_index,
                    BOND_ROLE_POLYMER,
                ) != tuple(expected_polymer_edges):
                    raise ValueError(f"mmCIF cannot reconstruct polymer bonds for entity {entity_id!r}")
            entity_category = "polymer"
        entity_columns["id"].append(entity_id)
        entity_columns["type"].append(entity_category)

        if type_code != ENTITY_MOLECULE:
            entity_poly_columns["entity_id"].append(entity_id)
            entity_poly_columns["type"].append(entity_type_name[type_code])
            for local_index, candidate_ids in enumerate(
                component_ids_by_position,
                1,
            ):
                if len(candidate_ids) > 1:
                    for instance_index in instances_by_entity[entity_index]:
                        occurrence = instance_components(instance_index)[local_index - 1]
                        atom_start = int(component_atom_ptr[occurrence])
                        atom_stop = int(component_atom_ptr[occurrence + 1])
                        if not bool(coordinate_mask[atom_start:atom_stop].any()):
                            raise ValueError("mmCIF cannot select an unobserved microheterogeneous polymer component")
                for candidate in candidate_ids:
                    entity_poly_seq_columns["entity_id"].append(entity_id)
                    entity_poly_seq_columns["num"].append(local_index)
                    entity_poly_seq_columns["mon_id"].append(candidate)
                    entity_poly_seq_columns["hetero"].append("y" if len(candidate_ids) > 1 else "n")
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    author_number = source_value(residue_id, component_index)
                    monomer = component_id(component_index)
                    asym_id = label_asym_by_instance[instance_index]
                    poly_scheme_values = (
                        asym_id,
                        entity_id,
                        local_index + 1,
                        monomer,
                        local_index + 1,
                        author_number or "?",
                        author_number or "?",
                        monomer,
                        monomer,
                        asym_id,
                        source_value(insertion_code, component_index) or "?",
                        ("y" if len(component_ids_by_position[local_index]) > 1 else "n"),
                    )
                    for name, value in zip(poly_scheme_columns, poly_scheme_values):
                        poly_scheme_columns[name].append(value)
        elif int(entity_molecule_type[entity_index]) == MOLECULE_BRANCHED:
            branch_columns["entity_id"].append(entity_id)
            branch_columns["type"].append("oligosaccharide")
            for local_index, component_index in enumerate(components, 1):
                branch_list_columns["entity_id"].append(entity_id)
                branch_list_columns["num"].append(local_index)
                branch_list_columns["comp_id"].append(component_id(component_index))
                branch_list_columns["hetero"].append("n")

            link_id = 1
            for bond_index in range(edge_index.shape[1]):
                if int(bond_role[bond_index]) != BOND_ROLE_ENTITY:
                    continue
                atom_a = int(edge_index[0, bond_index])
                atom_b = int(edge_index[1, bond_index])
                component_a = atom_component[atom_a]
                component_b = atom_component[atom_b]
                if (
                    component_instance[component_a] != first_instance
                    or component_instance[component_b] != first_instance
                ):
                    continue
                local_a = component_a - component_start
                local_b = component_b - component_start
                if local_a == local_b:
                    continue
                branch_link_values = (
                    entity_id,
                    link_id,
                    local_a + 1,
                    component_id(component_a),
                    atom_id(atom_a),
                    branch_leaving_by_endpoint.get((bond_index, 0), "?"),
                    local_b + 1,
                    component_id(component_b),
                    atom_id(atom_b),
                    branch_leaving_by_endpoint.get((bond_index, 1), "?"),
                    connection_order_name(bond_index),
                )
                for name, value in zip(branch_link_columns, branch_link_values):
                    branch_link_columns[name].append(value)
                link_id += 1

            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                instance_component_start = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                monomer_counts = {
                    component_id(index): sum(component_id(other) == component_id(index) for other in components)
                    for index in components
                }
                for local_index in range(len(components)):
                    component_index = instance_component_start + local_index
                    source_author_number = source_value(
                        residue_id,
                        component_index,
                    )
                    monomer = component_id(component_index)
                    atom_start = int(component_atom_ptr[component_index])
                    atom_stop = int(component_atom_ptr[component_index + 1])
                    if (
                        source_author_number is None
                        and monomer_counts[monomer] > 1
                        and bool(coordinate_mask[atom_start:atom_stop].any())
                    ):
                        raise ValueError(
                            "mmCIF cannot identify an observed repeated "
                            "branched-entity monomer without an author "
                            "sequence number"
                        )
                    auth_number = source_author_number or "?"
                    branch_scheme_values = (
                        asym_id,
                        entity_id,
                        monomer,
                        local_index + 1,
                        asym_id,
                        monomer,
                        auth_number,
                        source_value(insertion_code, component_index) or "?",
                        asym_id,
                        monomer,
                        auth_number,
                    )
                    for name, value in zip(branch_scheme_columns, branch_scheme_values):
                        branch_scheme_columns[name].append(value)
        else:
            monomer = component_id(components[0])
            nonpoly_columns["entity_id"].append(entity_id)
            nonpoly_columns["name"].append(monomer)
            nonpoly_columns["comp_id"].append(monomer)
            for instance_index in range(instance_count):
                if int(instance_entity[instance_index]) != entity_index:
                    continue
                component_index = int(instance_component_ptr[instance_index])
                asym_id = label_asym_by_instance[instance_index]
                author_number = source_value(residue_id, component_index) or "?"
                nonpoly_scheme_values = (
                    asym_id,
                    entity_id,
                    monomer,
                    1,
                    author_number,
                    author_number,
                    monomer,
                    monomer,
                    asym_id,
                    source_value(insertion_code, component_index) or "?",
                )
                for name, value in zip(nonpoly_scheme_columns, nonpoly_scheme_values):
                    nonpoly_scheme_columns[name].append(value)

    chem_comp_columns: dict[str, list[object]] = {
        "id": [],
        "type": [],
        "mon_nstd_parent_comp_id": [],
    }
    chem_atom_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id": [],
        "type_symbol": [],
        "charge": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
        "model_Cartn_x": [],
        "model_Cartn_y": [],
        "model_Cartn_z": [],
    }
    chem_bond_columns: dict[str, list[object]] = {
        "comp_id": [],
        "atom_id_1": [],
        "atom_id_2": [],
        "value_order": [],
        "pdbx_aromatic_flag": [],
        "pdbx_stereo_config": [],
    }
    component_bond_indices: dict[int, list[int]] = {}
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] == component_index:
            component_bond_indices.setdefault(component_index, []).append(bond_index)

    def component_definition(index: int) -> tuple[object, ...]:
        start = int(component_atom_ptr[index])
        stop = int(component_atom_ptr[index + 1])
        atoms = tuple(
            (
                atom_id(atom_index),
                element(atom_index),
                (int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else None),
                (bool(atom_aromatic[atom_index]) if bool(atom_aromatic_known[atom_index]) else None),
                (system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else None),
                (
                    tuple(float(value) for value in reference_coordinates[atom_index])
                    if bool(reference_coordinate_mask[atom_index])
                    else None
                ),
            )
            for atom_index in range(start, stop)
        )
        bonds = tuple(
            (
                atom_id(int(edge_index[0, bond_index])),
                atom_id(int(edge_index[1, bond_index])),
                *component_bond_values(bond_index),
                (system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else None),
            )
            for bond_index in component_bond_indices.get(index, ())
        )
        return bool(component_topology_complete[index]), atoms, bonds

    def known_component_signatures(
        component_index: int,
        expected: tuple[object, ...],
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        actual = component_signatures[component_index]
        actual_atoms = cast(tuple[tuple[object, ...], ...], actual[1])
        expected_atoms = cast(tuple[tuple[object, ...], ...], expected[1])
        if len(actual_atoms) != len(expected_atoms):
            return actual, expected

        normalized_atoms = []
        start = int(component_atom_ptr[component_index])
        for offset, (actual_atom, expected_atom) in enumerate(zip(actual_atoms, expected_atoms)):
            normalized = list(expected_atom)
            if actual_atom[:2] == expected_atom[:2]:
                observed = bool(coordinate_mask[start + offset])
                if actual_atom[2] != expected_atom[2]:
                    if not observed or actual_atom[1] != 1 or actual_atom[2] not in {2, 3}:
                        raise ValueError("mmCIF cannot encode this isotope difference for a known component")
                    normalized[2] = actual_atom[2]
                if actual_atom[3] != expected_atom[3]:
                    if not observed or actual_atom[3] is None:
                        raise ValueError("mmCIF cannot encode this formal-charge difference for a known component")
                    normalized[3] = actual_atom[3]
            normalized_atoms.append(tuple(normalized))
        return actual, (expected[0], tuple(normalized_atoms), expected[2])

    components_by_id: dict[str, list[int]] = {}
    for component_index in range(component_count):
        components_by_id.setdefault(component_id(component_index), []).append(component_index)
    chem_comp_type_by_id: dict[str, str] = {}
    for name, component_indices in components_by_id.items():
        polymer_type = get_builtin_component_polymer_type(name)
        component_types: set[str]
        if polymer_type == "protein":
            component_types = {"L-peptide linking"}
        elif polymer_type == "rna":
            component_types = {"RNA linking"}
        elif polymer_type == "dna":
            component_types = {"DNA linking"}
        else:
            component_types = set()
            for component_index in component_indices:
                instance_index = component_instance[component_index]
                entity_index = int(instance_entity[instance_index])
                type_code = int(entity_type[entity_index])
                if type_code == ENTITY_PROTEIN:
                    component_types.add("L-peptide linking")
                elif type_code == ENTITY_RNA:
                    component_types.add("RNA linking")
                elif type_code == ENTITY_DNA:
                    component_types.add("DNA linking")
                elif type_code == ENTITY_MOLECULE:
                    molecule_type = int(entity_molecule_type[entity_index])
                    component_types.add("saccharide" if molecule_type == MOLECULE_BRANCHED else "non-polymer")
                else:
                    component_types.add("other")
        chem_comp_type = next(iter(component_types)) if len(component_types) == 1 else "other"
        chem_comp_type_by_id[name] = chem_comp_type
        chem_comp_columns["id"].append(name)
        chem_comp_columns["type"].append(chem_comp_type)
        parent_lists = {structure.component_parents(index) for index in component_indices}
        if len(parent_lists) > 1:
            raise ValueError(f"component {name!r} has conflicting parent declarations across occurrences")
        parents = parent_lists.pop()
        chem_comp_columns["mon_nstd_parent_comp_id"].append(",".join(parents) if parents else "?")
    embedded_component_by_id: dict[str, int] = {}
    component_signatures = _graph_component_signatures(system)
    polymer_types: dict[int, PolymerType] = {
        ENTITY_PROTEIN: "protein",
        ENTITY_RNA: "rna",
        ENTITY_DNA: "dna",
        ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
        ENTITY_PNA: "pna",
    }

    expected_definitions: dict[tuple[str, int, bool, bool, tuple[str, ...]], tuple[object, ...]] = {}
    for name, component_indices in components_by_id.items():
        try:
            authoritative = get_component(name)
        except KeyError:
            first = component_indices[0]
            definition = component_definition(first)
            if any(component_definition(component_index) != definition for component_index in component_indices[1:]):
                raise ValueError(f"mmCIF cannot encode conflicting realized definitions for component ID {name!r}")
            embedded_component_by_id[name] = first
            continue

        for component_index in component_indices:
            instance_index = component_instance[component_index]
            component_start = int(instance_component_ptr[instance_index])
            component_stop = int(instance_component_ptr[instance_index + 1])
            has_previous = component_index > component_start and (component_index - 1, component_index) in polymer_pairs
            has_next = component_index + 1 < component_stop and (component_index, component_index + 1) in polymer_pairs
            entity_index = int(instance_entity[instance_index])
            type_code = int(entity_type[entity_index])
            leaving_atoms = tuple(
                sorted(
                    branch_leaving_by_component.get(component_index, frozenset())
                    | assembly_leaving_by_component.get(component_index, frozenset())
                )
            )
            key = (
                name,
                type_code,
                has_previous,
                has_next,
                leaving_atoms,
            )
            expected = expected_definitions.get(key)
            if expected is None:
                expected_component = authoritative
                if leaving_atoms:
                    expected_component = _remove_component_atoms(
                        expected_component,
                        set(leaving_atoms),
                    )
                expected = _realized_component_signature(
                    expected_component,
                    None if type_code == ENTITY_MOLECULE else polymer_types[type_code],
                    has_previous=has_previous,
                    has_next=has_next,
                )
                expected_definitions[key] = expected
            actual, normalized_expected = known_component_signatures(
                component_index,
                expected,
            )
            if actual != normalized_expected:
                raise ValueError(f"component ID {name!r} does not match its authoritative definition")

    for name in embedded_component_by_id:
        if chem_comp_type_by_id[name] == "other" and any(
            name_component in pair for name_component in components_by_id[name] for pair in polymer_pairs
        ):
            raise ValueError(
                f"mmCIF cannot reconstruct polymer attachments for component {name!r} with chem_comp.type other"
            )
        for component_index in components_by_id[name]:
            start = int(component_atom_ptr[component_index])
            stop = int(component_atom_ptr[component_index + 1])
            decoded_topology_complete = stop - start == 1 or bool(component_bond_indices.get(component_index))
            if bool(component_topology_complete[component_index]) != decoded_topology_complete:
                raise ValueError("mmCIF cannot encode this unknown component's topology completeness")
            isotope_representable = (atomic_number[start:stop] == 1) & torch.isin(
                isotope[start:stop],
                isotope.new_tensor((2, 3)),
            )
            if bool((isotope_known[start:stop] & ~isotope_representable).any()):
                raise ValueError(
                    "mmCIF cannot encode explicit isotopes other than deuterium and tritium for an unknown component"
                )
            for field in (
                "radical_electrons_known",
                "explicit_hydrogens_known",
                "implicit_hydrogens_known",
                "implicit_hydrogens_allowed_known",
            ):
                if bool(getattr(system, field)[start:stop].any()):
                    raise ValueError(
                        f"mmCIF cannot encode known {field.removesuffix('_known')} for an unknown component"
                    )
            for edge in component_bond_indices.get(component_index, ()):
                if (
                    int(bond_kind[edge]) != BOND_KIND_COVALENT
                    or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
                    or bool((system.bond_stereo_index[:, edge] >= 0).any())
                ):
                    raise ValueError(
                        "mmCIF chem_comp_bond cannot encode directed, dative, or stereo-defining component bonds"
                    )

    for edge in range(edge_index.shape[1]):
        role = int(bond_role[edge])
        if role == BOND_ROLE_ENTITY and (
            int(bond_kind[edge]) != BOND_KIND_COVALENT
            or int(bond_direction[edge]) != BOND_DIRECTION_UNDIRECTED
            or int(bond_stereo[edge]) >= 0
            or bool((system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF branched-entity links cannot encode directed, dative, or stereochemical bonds")
        if role >= BOND_ROLE_POLYMER and (
            int(bond_stereo[edge]) >= 0 or bool((system.bond_stereo_index[:, edge] >= 0).any())
        ):
            raise ValueError("mmCIF struct_conn cannot encode bond stereochemistry")
    for name, component_index in embedded_component_by_id.items():
        atom_start = int(component_atom_ptr[component_index])
        atom_stop = int(component_atom_ptr[component_index + 1])
        for atom_index in range(atom_start, atom_stop):
            reference = reference_coordinates[atom_index]
            chem_atom_values = (
                name,
                atom_id(atom_index),
                element(atom_index),
                int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
                ("?" if not bool(atom_aromatic_known[atom_index]) else "Y" if bool(atom_aromatic[atom_index]) else "N"),
                format_chemical_stereo(
                    system.symbols[int(atom_stereo[atom_index])] if int(atom_stereo[atom_index]) >= 0 else None,
                    category="atom",
                ),
                float(reference[0]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[1]) if bool(reference_coordinate_mask[atom_index]) else "?",
                float(reference[2]) if bool(reference_coordinate_mask[atom_index]) else "?",
            )
            for column, value in zip(chem_atom_columns, chem_atom_values):
                chem_atom_columns[column].append(value)

    emitted_component_bonds: set[tuple[str, str, str]] = set()
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        component_index = atom_component[atom_a]
        if atom_component[atom_b] != component_index:
            continue
        name = component_id(component_index)
        if embedded_component_by_id.get(name) != component_index:
            continue
        component_bond_key = (name, atom_id(atom_a), atom_id(atom_b))
        if component_bond_key in emitted_component_bonds:
            continue
        emitted_component_bonds.add(component_bond_key)
        order_value, aromatic_value = component_bond_values(bond_index)
        chem_bond_values = (
            name,
            atom_id(atom_a),
            atom_id(atom_b),
            order_value,
            aromatic_value,
            format_chemical_stereo(
                system.symbols[int(bond_stereo[bond_index])] if int(bond_stereo[bond_index]) >= 0 else None,
                category="bond",
            ),
        )
        for column, value in zip(chem_bond_columns, chem_bond_values):
            chem_bond_columns[column].append(value)

    atom_site_columns: dict[str, list[object]] = {
        "group_PDB": [],
        "id": [],
        "type_symbol": [],
        "label_atom_id": [],
        "label_alt_id": [],
        "label_comp_id": [],
        "label_asym_id": [],
        "label_entity_id": [],
        "label_seq_id": [],
        "pdbx_PDB_ins_code": [],
        "Cartn_x": [],
        "Cartn_y": [],
        "Cartn_z": [],
        "occupancy": [],
        "B_iso_or_equiv": [],
        "pdbx_formal_charge": [],
        "auth_seq_id": [],
        "auth_comp_id": [],
        "auth_asym_id": [],
        "auth_atom_id": [],
        "pdbx_PDB_model_num": [],
    }
    provenance_columns: dict[str, list[object]] = {
        "label_asym_id": [],
        "component_index": [],
        "label_atom_id": [],
    }
    endpoint_identity: dict[int, tuple[object, ...]] = {}
    serial = 1
    for atom_index in range(atom_count):
        component_index = atom_component[atom_index]
        instance_index = component_instance[component_index]
        entity_index = int(instance_entity[instance_index])
        local_component = component_index - int(instance_component_ptr[instance_index])
        polymer = int(entity_type[entity_index]) != ENTITY_MOLECULE
        asym_id = label_asym_by_instance[instance_index]
        monomer = component_id(component_index)
        source_author_number = source_value(residue_id, component_index)
        author_number = source_author_number or "?"
        code = source_value(insertion_code, component_index)
        label_sequence: object = local_component + 1 if polymer else "."
        endpoint_identity[atom_index] = (
            asym_id,
            label_sequence,
            monomer,
            atom_id(atom_index),
            asym_id,
            author_number,
            monomer,
            atom_id(atom_index),
            code or "?",
        )
        if bool(source_formal_charge_mask[atom_index]):
            provenance_values = (asym_id, local_component, atom_id(atom_index))
            for column, value in zip(provenance_columns, provenance_values, strict=True):
                provenance_columns[column].append(value)
        if not bool(coordinate_mask[atom_index]):
            continue
        xyz = coordinates[atom_index]
        if not bool(torch.isfinite(xyz).all()):
            raise ValueError("mmCIF cannot represent non-finite observed coordinates")
        if bool(occupancy_mask[atom_index]):
            atom_occupancy: float | str = float(occupancy[atom_index])
            if not np.isfinite(atom_occupancy):
                raise ValueError("mmCIF cannot represent non-finite occupancy")
        else:
            atom_occupancy = "."
        if bool(b_factor_mask[atom_index]):
            atom_b_factor: float | str = float(b_factor[atom_index])
            if not np.isfinite(atom_b_factor):
                raise ValueError("mmCIF cannot represent non-finite B factor")
        else:
            atom_b_factor = "."
        atom_site_values = (
            "ATOM" if polymer else "HETATM",
            serial,
            element(atom_index),
            atom_id(atom_index),
            ".",
            monomer,
            asym_id,
            entity_id_by_index[entity_index],
            label_sequence,
            code or "?",
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            atom_occupancy,
            atom_b_factor,
            int(formal_charge[atom_index]) if bool(formal_charge_known[atom_index]) else "?",
            author_number,
            monomer,
            asym_id,
            atom_id(atom_index),
            1,
        )
        for column, value in zip(atom_site_columns, atom_site_values):
            atom_site_columns[column].append(value)
        serial += 1

    conn_columns: dict[str, list[object]] = {
        "id": [],
        "conn_type_id": [],
        "pdbx_leaving_atom_flag": [],
        "pdbx_ptnr1_leaving_atom_id": [],
        "pdbx_ptnr2_leaving_atom_id": [],
        "ptnr1_label_asym_id": [],
        "ptnr1_label_seq_id": [],
        "ptnr1_label_comp_id": [],
        "ptnr1_label_atom_id": [],
        "ptnr2_label_asym_id": [],
        "ptnr2_label_seq_id": [],
        "ptnr2_label_comp_id": [],
        "ptnr2_label_atom_id": [],
        "ptnr1_auth_asym_id": [],
        "ptnr1_auth_seq_id": [],
        "ptnr1_auth_comp_id": [],
        "ptnr1_auth_atom_id": [],
        "ptnr2_auth_asym_id": [],
        "ptnr2_auth_seq_id": [],
        "ptnr2_auth_comp_id": [],
        "ptnr2_auth_atom_id": [],
        "pdbx_ptnr1_PDB_ins_code": [],
        "pdbx_ptnr2_PDB_ins_code": [],
        "ptnr1_role": [],
        "ptnr2_role": [],
        "ptnr1_symmetry": [],
        "ptnr2_symmetry": [],
        "pdbx_value_order": [],
    }
    connection_id = 1
    for bond_index in range(edge_index.shape[1]):
        if int(bond_role[bond_index]) != BOND_ROLE_ASSEMBLY:
            continue
        atom_a = int(edge_index[0, bond_index])
        atom_b = int(edge_index[1, bond_index])
        kind = int(bond_kind[bond_index])
        direction = int(bond_direction[bond_index])
        if kind == BOND_KIND_DATIVE:
            if direction == BOND_DIRECTION_ENDPOINT_0_DONOR:
                roles = ("donor", "acceptor")
            elif direction == BOND_DIRECTION_ENDPOINT_1_DONOR:
                roles = ("acceptor", "donor")
            else:
                raise ValueError("mmCIF struct_conn requires a donor direction for a dative bond")
            conn_type = "metalc"
        elif kind == BOND_KIND_COVALENT:
            if direction != BOND_DIRECTION_UNDIRECTED:
                raise ValueError("mmCIF struct_conn cannot encode a directed covalent bond")
            roles = ("?", "?")
            conn_type = "covale"
        else:
            raise ValueError(f"unknown canonical bond kind {kind}")
        value_order = connection_order_name(bond_index)
        if value_order == "arom":
            raise ValueError("mmCIF struct_conn pdbx_value_order does not support aromatic bonds")
        first_endpoint = endpoint_identity[atom_a]
        second_endpoint = endpoint_identity[atom_b]
        first_leaving = assembly_leaving_by_endpoint.get((bond_index, 0))
        second_leaving = assembly_leaving_by_endpoint.get((bond_index, 1))
        leaving_count = int(first_leaving is not None) + int(second_leaving is not None)
        leaving_flag = ("none", "one", "both")[leaving_count]
        connection_values = (
            connection_id,
            conn_type,
            leaving_flag,
            "?" if first_leaving is None else first_leaving[0],
            "?" if second_leaving is None else second_leaving[0],
            *first_endpoint[:4],
            *second_endpoint[:4],
            *first_endpoint[4:8],
            *second_endpoint[4:8],
            first_endpoint[8],
            second_endpoint[8],
            *roles,
            "1_555",
            "1_555",
            value_order,
        )
        for column, value in zip(conn_columns, connection_values):
            conn_columns[column].append(value)
        connection_id += 1

    categories: dict[str, CIFCategory | str] = {
        "entity": CIFCategory(entity_columns),
        "struct_asym": CIFCategory(struct_asym_columns),
        "ofoldx_provenance": CIFCategory({"formal_charge_source_complete": ["Y"]}),
    }
    if provenance_columns["label_asym_id"]:
        categories["ofoldx_source_formal_charge"] = CIFCategory(provenance_columns)
    resolution_rows: dict[int, list[float]] = {}
    for value, kind in zip(resolution.tolist(), resolution_kind.tolist(), strict=True):
        if kind not in {
            RESOLUTION_KIND_REFINEMENT,
            RESOLUTION_KIND_REFLECTIONS,
            RESOLUTION_KIND_EM_RECONSTRUCTION,
        }:
            raise ValueError(f"mmCIF cannot encode experimental-resolution kind {kind}")
        resolution_rows.setdefault(kind, []).append(value)
    if resolution_rows:
        categories["entry"] = CIFCategory({"id": ["ofoldx"]})
    refinement = resolution_rows.get(RESOLUTION_KIND_REFINEMENT, [])
    if refinement:
        categories["refine"] = CIFCategory(
            {
                "entry_id": ["ofoldx"] * len(refinement),
                "pdbx_refine_id": [f"OFOLDX {index}" for index in range(1, len(refinement) + 1)],
                "ls_d_res_high": refinement,
            }
        )
    reflections = resolution_rows.get(RESOLUTION_KIND_REFLECTIONS, [])
    if reflections:
        categories["reflns"] = CIFCategory(
            {
                "pdbx_ordinal": list(range(1, len(reflections) + 1)),
                "entry_id": ["ofoldx"] * len(reflections),
                "d_resolution_high": reflections,
            }
        )
    reconstructions = resolution_rows.get(RESOLUTION_KIND_EM_RECONSTRUCTION, [])
    if reconstructions:
        categories["em_3d_reconstruction"] = CIFCategory(
            {
                "id": list(range(1, len(reconstructions) + 1)),
                "entry_id": ["ofoldx"] * len(reconstructions),
                "resolution": reconstructions,
            }
        )
    # A structure that resolves no atom still states its chemistry through the declaration
    # categories below, and mmCIF writes an absent loop rather than an empty one.
    if tuple(atom_site_columns.values())[0]:
        categories["atom_site"] = CIFCategory(atom_site_columns)
    for name, columns in (
        ("entity_poly", entity_poly_columns),
        ("entity_poly_seq", entity_poly_seq_columns),
        ("pdbx_poly_seq_scheme", poly_scheme_columns),
        ("pdbx_entity_branch", branch_columns),
        ("pdbx_entity_branch_list", branch_list_columns),
        ("pdbx_entity_branch_link", branch_link_columns),
        ("pdbx_branch_scheme", branch_scheme_columns),
        ("pdbx_entity_nonpoly", nonpoly_columns),
        ("pdbx_nonpoly_scheme", nonpoly_scheme_columns),
        ("chem_comp", chem_comp_columns),
        ("chem_comp_atom", chem_atom_columns),
        ("chem_comp_bond", chem_bond_columns),
        ("struct_conn", conn_columns),
    ):
        if columns and tuple(columns.values())[0]:
            categories[name] = CIFCategory(columns)
    CIFFile({"ofoldx": CIFBlock(categories)}).write(str(path))
