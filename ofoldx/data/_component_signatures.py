"""Exact component graph value comparison shared by codecs."""

from __future__ import annotations

from .molecular import (
    BOND_ROLE_COMPONENT,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerAttachment,
    PolymerSequence,
    PolymerType,
)


def _graph_component_signatures(graph: MolecularSystem) -> tuple[tuple[object, ...], ...]:
    r"""Exact value signatures of every realized component subgraph."""

    component_count = int(graph.component_atom_ptr.numel()) - 1
    component_by_atom = [0] * int(graph.atom_symbol.numel())
    atoms_by_component: list[tuple[tuple[object, ...], ...]] = []
    for component_index in range(component_count):
        start = int(graph.component_atom_ptr[component_index])
        stop = int(graph.component_atom_ptr[component_index + 1])
        component_by_atom[start:stop] = [component_index] * (stop - start)
        atoms_by_component.append(
            tuple(
                (
                    graph.symbols[int(graph.atom_symbol[atom])],
                    int(graph.atomic_number[atom]),
                    int(graph.isotope[atom]) if bool(graph.isotope_known[atom]) else None,
                    int(graph.formal_charge[atom]) if bool(graph.formal_charge_known[atom]) else None,
                    int(graph.radical_electrons[atom]) if bool(graph.radical_electrons_known[atom]) else None,
                    int(graph.explicit_hydrogens[atom]) if bool(graph.explicit_hydrogens_known[atom]) else None,
                    int(graph.implicit_hydrogens[atom]) if bool(graph.implicit_hydrogens_known[atom]) else None,
                    (
                        bool(graph.implicit_hydrogens_allowed[atom])
                        if bool(graph.implicit_hydrogens_allowed_known[atom])
                        else None
                    ),
                    bool(graph.atom_aromatic[atom]) if bool(graph.atom_aromatic_known[atom]) else None,
                    None if int(graph.atom_stereo[atom]) < 0 else graph.symbols[int(graph.atom_stereo[atom])],
                    (
                        tuple(float(value) for value in graph.reference_coordinates[atom])
                        if bool(graph.reference_coordinate_mask[atom])
                        else None
                    ),
                )
                for atom in range(start, stop)
            )
        )

    edges_by_component: list[list[tuple[object, ...]]] = [[] for _ in range(component_count)]
    for edge in range(graph.edge_index.shape[1]):
        if int(graph.bond_role[edge]) != BOND_ROLE_COMPONENT:
            continue
        atom_a = int(graph.edge_index[0, edge])
        atom_b = int(graph.edge_index[1, edge])
        component_index = component_by_atom[atom_a]
        if component_by_atom[atom_b] != component_index:
            continue
        start = int(graph.component_atom_ptr[component_index])
        stereo = int(graph.bond_stereo[edge])
        stereo_atoms = graph.bond_stereo_index[:, edge]
        edges_by_component[component_index].append(
            (
                atom_a - start,
                atom_b - start,
                float(graph.bond_order[edge]) if bool(graph.bond_order_known[edge]) else None,
                bool(graph.bond_aromatic[edge]) if bool(graph.bond_aromatic_known[edge]) else None,
                int(graph.bond_kind[edge]),
                int(graph.bond_direction[edge]),
                None if stereo < 0 else graph.symbols[stereo],
                tuple(-1 if int(atom) < 0 else int(atom) - start for atom in stereo_atoms),
            )
        )
    return tuple(
        (
            bool(graph.component_topology_complete[index]),
            atoms_by_component[index],
            tuple(edges_by_component[index]),
        )
        for index in range(component_count)
    )


def _realized_component_signature(
    component: ChemicalComponent,
    polymer_type: PolymerType | None,
    *,
    has_previous: bool = False,
    has_next: bool = False,
) -> tuple[object, ...]:
    r"""Signature after realizing a component in its stated linear-polymer context."""

    if polymer_type is None:
        entity: Molecule | PolymerSequence = Molecule((component,))
        component_index = 0
    else:
        left = ChemicalComponent(
            "__LEFT__",
            (ChemicalAtom("X", "C"),),
            bonds=frozenset(),
            next_attachment=PolymerAttachment("X"),
        )
        right = ChemicalComponent(
            "__RIGHT__",
            (ChemicalAtom("X", "C"),),
            bonds=frozenset(),
            previous_attachment=PolymerAttachment("X"),
        )
        components = (
            *((left,) if has_previous else ()),
            component,
            *((right,) if has_next else ()),
        )
        entity = PolymerSequence(polymer_type, components)
        component_index = int(has_previous)
    expected = MolecularSystem.from_entities({"expected": entity}, {"expected": "expected"})
    return _graph_component_signatures(expected)[component_index]
