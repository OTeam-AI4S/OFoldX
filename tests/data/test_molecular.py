from __future__ import annotations

import pytest
import torch

from ofoldx.data import (
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    MOLECULE_BRANCHED,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerAttachment,
    PolymerSequence,
    SystemBond,
    component_boundary_atoms,
    copy_components,
    disjoint_union,
    form_bond,
    instance_subgraph,
    replace_components,
    water_component_mask,
)


def _residue(
    component_id: str,
    side_chain: str | None = None,
    *,
    stereo: str | None = None,
) -> ChemicalComponent:
    atoms = [
        ChemicalAtom("N", "N", stereo=stereo),
        ChemicalAtom("H2", "H"),
        ChemicalAtom("CA", "C"),
        ChemicalAtom("C", "C", stereo=stereo),
        ChemicalAtom("OXT", "O"),
    ]
    bonds = [
        ChemicalBond("N", "CA"),
        ChemicalBond("CA", "C"),
        ChemicalBond("C", "OXT"),
    ]
    if side_chain is not None:
        atoms.append(ChemicalAtom(side_chain, "C"))
        bonds.append(ChemicalBond("CA", side_chain))
    return ChemicalComponent(
        component_id,
        tuple(atoms),
        frozenset(bonds),
        previous_attachment=PolymerAttachment("N", frozenset({"H2"})),
        next_attachment=PolymerAttachment("C", frozenset({"OXT"})),
    )


class TestConstructionValues:
    def test_sequence_and_molecule_keep_exact_chemistry(self) -> None:
        sequence = PolymerSequence.from_one_letter("AG")
        assert sequence.to_one_letter() == "AG"
        assert tuple(component.component_id for component in sequence.components) == (
            "ALA",
            "GLY",
        )

        partial = ChemicalComponent(
            "PART",
            (ChemicalAtom("C1", "C", formal_charge=None),),
            None,
        )
        complete = ChemicalComponent(
            "COMP",
            (ChemicalAtom("C1", "C", formal_charge=0),),
            frozenset(),
        )
        molecule = Molecule(
            (partial, complete),
            frozenset({MoleculeBond(0, "C1", 1, "C1")}),
        )
        assert molecule.components[0].bonds is None
        assert molecule.components[1].bonds == frozenset()
        assert next(iter(molecule.bonds)).order == 1.0

    def test_component_hydrogen_projection_preserves_defined_stereo(self) -> None:
        deuterium = ChemicalAtom("D1", "D")
        assert (deuterium.element, deuterium.isotope) == ("H", 2)

        component = ChemicalComponent(
            "D",
            (
                ChemicalAtom("C1", "C"),
                ChemicalAtom("C2", "C"),
                deuterium,
            ),
            frozenset(
                {
                    ChemicalBond("C1", "C2", stereo="E"),
                    ChemicalBond("C1", "D1"),
                }
            ),
        )
        heavy = component.without_hydrogens()
        assert tuple(atom.atom_id for atom in heavy.atoms) == ("C1", "C2")
        assert heavy.bonds == frozenset({ChemicalBond("C1", "C2", stereo="E")})

    def test_component_rejects_an_unresolved_bond_endpoint(self) -> None:
        with pytest.raises(ValueError, match="unknown atom"):
            ChemicalComponent(
                "BAD",
                (ChemicalAtom("C1", "C"),),
                frozenset({ChemicalBond("C1", "missing")}),
            )


class TestMolecularSystem:
    def test_materialization_exposes_one_canonical_tensor_graph(self) -> None:
        protein = PolymerSequence(
            "protein",
            (_residue("R0", "CB"), _residue("R1")),
        )
        ligand = Molecule(
            (
                ChemicalComponent(
                    "LIG",
                    (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
                    frozenset({ChemicalBond("C1", "O1")}),
                ),
            ),
            molecule_type="branched",
        )
        system = MolecularSystem.from_entities(
            {"protein": protein, "ligand": ligand},
            {"A": "protein", "B": "protein", "L": "ligand"},
            (
                SystemBond(
                    AtomSite("A", 0, "CB"),
                    AtomSite("L", 0, "C1"),
                ),
            ),
        )

        assert system.system_entity_ptr.tolist() == [0, 2]
        assert system.system_instance_ptr.tolist() == [0, 3]
        assert system.instance_component_ptr.tolist() == [0, 2, 4, 5]
        assert system.entity_id(system.entity_index("protein")) == "protein"
        assert system.entity_molecule_type[system.entity_index("ligand")] == MOLECULE_BRANCHED
        assert system.instance_id(system.instance_index("B")) == "B"
        component = system.component_index(("A", 1))
        assert system.component_reference(component) == ("A", 1)
        atom = system.atom_index(AtomSite("A", 0, "CB"))
        assert system.atom_site(atom) == AtomSite("A", 0, "CB")
        assert set(system.bond_role.tolist()) == {
            BOND_ROLE_COMPONENT,
            BOND_ROLE_POLYMER,
            BOND_ROLE_ASSEMBLY,
        }
        assert AtomSite("A", 0, "OXT") not in {system.atom_site(index) for index in range(system.atom_symbol.numel())}
        assert AtomSite("A", 1, "H2") not in {system.atom_site(index) for index in range(system.atom_symbol.numel())}

        double = system.double()
        assert double.reference_coordinates.dtype == torch.float64
        assert double.bond_order.dtype == torch.float64
        assert double.edge_index.dtype == torch.int64
        assert double.component_topology_complete.dtype == torch.bool

    def test_duplicate_physical_edge_requires_identical_leaving_atoms(self) -> None:
        donor = ChemicalComponent(
            "DON",
            (ChemicalAtom("E", "C"), ChemicalAtom("L1", "H"), ChemicalAtom("L2", "H")),
            frozenset({ChemicalBond("E", "L1"), ChemicalBond("E", "L2")}),
        )
        acceptor = ChemicalComponent("ACC", (ChemicalAtom("X", "N"),), frozenset())
        entities = {"donor": Molecule((donor,)), "acceptor": Molecule((acceptor,))}
        instances = {"A": "donor", "B": "acceptor"}

        with pytest.raises(ValueError, match="conflicting leaving atoms for physical edge"):
            MolecularSystem.from_entities(
                entities,
                instances,
                (
                    SystemBond(AtomSite("A", 0, "E"), AtomSite("B", 0, "X"), leaving_a=frozenset({"L1"})),
                    SystemBond(AtomSite("A", 0, "E"), AtomSite("B", 0, "X"), leaving_a=frozenset({"L2"})),
                ),
            )

    def test_instance_values_preserve_one_entity_row(self) -> None:
        declared = PolymerSequence(
            "protein",
            (ChemicalComponent("ALA", (ChemicalAtom("CA", "C"),), frozenset()),),
        )
        alternate = PolymerSequence(
            "protein",
            (
                ChemicalComponent(
                    "GLY",
                    (ChemicalAtom("CA", "C"), ChemicalAtom("O", "O")),
                    frozenset({ChemicalBond("CA", "O")}),
                ),
            ),
        )
        system = MolecularSystem.from_entities(
            {"protein": declared},
            {"A": "protein", "B": "protein"},
            instance_values={"B": alternate},
        )

        assert system.entity_symbol.numel() == 1
        assert system.instance_entity.tolist() == [0, 0]
        assert [system.symbols[index] for index in system.component_symbol.tolist()] == ["ALA", "GLY"]
        assert system.component_atom_ptr.tolist() == [0, 1, 3]

    def test_water_mask_uses_the_declared_entity_category(self) -> None:
        oxygen = (ChemicalAtom("O", "O"),)
        system = MolecularSystem.from_entities(
            {
                "water": Molecule(
                    (ChemicalComponent("CUSTOM", oxygen, frozenset()),),
                    molecule_type="water",
                ),
                "ligand": Molecule(
                    (ChemicalComponent("HOH", oxygen, frozenset()),),
                ),
            },
            {"W": "water", "L": "ligand"},
        )
        assert water_component_mask(system).tolist() == [True, False]

    def test_batch_and_instance_subgraph_return_exact_axis_partitions(self) -> None:
        system = MolecularSystem.from_entities(
            {
                "protein": PolymerSequence(
                    "protein",
                    (_residue("R0"),),
                )
            },
            {"A": "protein", "B": "protein"},
        )
        selected, component_map, atom_map = instance_subgraph(
            system,
            torch.tensor([False, True]),
        )
        assert selected.instance_id(0) == "B"
        assert component_map.tolist() == [-1, 0]
        assert atom_map.tolist() == [-1] * 5 + list(range(5))

        batched = MolecularSystem.batch((selected, selected))
        assert batched.system_entity_ptr.tolist() == [0, 1, 2]
        assert batched.system_instance_ptr.tolist() == [0, 1, 2]
        first, second = batched.unbatch()
        assert first.instance_id(0) == second.instance_id(0) == "B"
        assert first.atom_symbol.tolist() == second.atom_symbol.tolist()


class TestGraphOperations:
    def test_substitution_and_bond_formation_return_atom_maps(self) -> None:
        atom = ChemicalComponent("ONE", (ChemicalAtom("C1", "C"),), frozenset())
        target = MolecularSystem.from_entities(
            {"target": Molecule((atom, atom))},
            {"T": "target"},
        )
        replacement = ChemicalComponent(
            "TWO",
            (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
            frozenset({ChemicalBond("C1", "O1")}),
        )
        replaced, target_map, replacement_map = replace_components(
            target,
            torch.tensor([0], dtype=torch.int64),
            (replacement,),
        )
        assert target_map.tolist() == [0, 2]
        assert replacement_map.tolist() == [0, 1]
        assert replaced.component_atom_ptr.tolist() == [0, 2, 3]

        copied, copied_target_map, copied_atom_index = copy_components(
            target,
            torch.tensor([1], dtype=torch.int64),
            replaced,
            torch.tensor([0], dtype=torch.int64),
        )
        assert copied_target_map.tolist() == [0, 1]
        assert copied_atom_index.tolist() == [[0, 1], [1, 2]]

        joined, old_to_new = form_bond(
            target,
            0,
            1,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
        assert old_to_new.tolist() == [0, 1]
        assert joined.edge_index.tolist() == [[0], [1]]
        assert joined.bond_role.tolist() == [BOND_ROLE_ASSEMBLY]

    def test_replace_components_preserves_an_unbonded_junction(self) -> None:
        nicked = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (_residue("R0"), _residue("R1")), linked=(False,))},
            {"A": "protein"},
        )
        replaced, _, _ = replace_components(
            nicked,
            torch.tensor([0, 1], dtype=torch.int64),
            (_residue("R2"), _residue("R3")),
        )
        sites = {replaced.atom_site(index) for index in range(replaced.atom_symbol.numel())}

        assert not bool((replaced.bond_role == BOND_ROLE_POLYMER).any())
        assert AtomSite("A", 0, "OXT") in sites
        assert AtomSite("A", 1, "H2") in sites

        linked = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (_residue("R0"), _residue("R1")))},
            {"A": "protein"},
        )
        bonded, _, _ = replace_components(
            linked,
            torch.tensor([0, 1], dtype=torch.int64),
            (_residue("R2"), _residue("R3")),
        )
        assert int((bonded.bond_role == BOND_ROLE_POLYMER).sum()) == 1

    def test_form_bond_invalidates_falsified_hydrogen_counts(self) -> None:
        methyl = ChemicalComponent(
            "CH3",
            (ChemicalAtom("C1", "C", explicit_hydrogens=0, implicit_hydrogens=3),),
            frozenset(),
        )
        system = MolecularSystem.from_entities(
            {"pair": Molecule((methyl, methyl))},
            {"M": "pair"},
        )
        bonded, _ = form_bond(
            system,
            0,
            1,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )

        assert bonded.implicit_hydrogens_known.tolist() == [False, False]
        assert bonded.explicit_hydrogens_known.tolist() == [True, True]
        assert bonded.explicit_hydrogens.tolist() == [0, 0]

    def test_copy_components_repeats_one_source_occurrence(self) -> None:
        placeholder = ChemicalComponent("ONE", (ChemicalAtom("C1", "C"),), frozenset())
        target = MolecularSystem.from_entities(
            {"target": Molecule((placeholder, placeholder))},
            {"T": "target"},
        )
        source_component = ChemicalComponent(
            "TWO",
            (ChemicalAtom("C1", "C"), ChemicalAtom("O1", "O")),
            frozenset({ChemicalBond("C1", "O1")}),
        )
        source = MolecularSystem.from_entities(
            {"source": Molecule((source_component,))},
            {"S": "source"},
        )

        copied, target_to_new, copied_atom_index = copy_components(
            target,
            torch.tensor([1, 0], dtype=torch.int64),
            source,
            torch.tensor([0, 0], dtype=torch.int64),
        )

        assert target_to_new.tolist() == [0, 2]
        assert copied_atom_index.tolist() == [[0, 1, 0, 1], [2, 3, 0, 1]]
        assert copied.component_atom_ptr.tolist() == [0, 2, 4]
        assert copied.edge_index.tolist() == [[0, 2], [1, 3]]
        assert copied.bond_role.tolist() == [BOND_ROLE_COMPONENT, BOND_ROLE_COMPONENT]

    def test_copy_components_preserves_unchanged_stereo(self) -> None:
        system = MolecularSystem.from_entities(
            {
                "protein": PolymerSequence(
                    "protein",
                    tuple(_residue(f"R{index}", stereo="R") for index in range(3)),
                )
            },
            {"A": "protein"},
        )
        copied, _, _ = copy_components(
            system,
            torch.arange(3, dtype=torch.int64),
            system,
            torch.arange(3, dtype=torch.int64),
        )

        assert torch.equal(copied.edge_index, system.edge_index)
        assert torch.equal(copied.atom_stereo, system.atom_stereo)

        isolated = MolecularSystem.from_entities(
            {"protein": PolymerSequence("protein", (_residue("T"),))},
            {"T": "protein"},
        )
        cut, _, _ = copy_components(
            isolated,
            torch.tensor([0], dtype=torch.int64),
            system,
            torch.tensor([1], dtype=torch.int64),
        )
        assert (cut.atom_stereo < 0).all()

    def test_disjoint_union_keeps_inputs_separate_and_reports_offsets(self) -> None:
        component = ChemicalComponent(
            "ION",
            (ChemicalAtom("X", "C"),),
            frozenset(),
        )
        first = MolecularSystem.from_entities(
            {"first": Molecule((component,))},
            {"A": "first"},
        )
        second = MolecularSystem.from_entities(
            {"second": Molecule((component,))},
            {"B": "second"},
        )
        union, component_maps, atom_maps = disjoint_union((first, second))

        assert union.system_entity_ptr.tolist() == [0, 2]
        assert union.system_instance_ptr.tolist() == [0, 2]
        assert [mapping.tolist() for mapping in component_maps] == [[0], [1]]
        assert [mapping.tolist() for mapping in atom_maps] == [[0], [1]]


    def test_component_boundaries(self) -> None:
        sequence = PolymerSequence("protein", (_residue("A"), _residue("B")))
        system = MolecularSystem.from_entities({"protein": sequence}, {"A": "protein"})
        first = system.atom_index(AtomSite("A", 0, "C"))
        second = system.atom_index(AtomSite("A", 1, "N"))

        assert component_boundary_atoms(system) == ((first,), (second,))

        atom_mask = torch.ones(system.atom_symbol.shape, dtype=torch.bool)
        atom_mask[second] = False
        assert component_boundary_atoms(system, atom_mask) == ((), ())
