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
    repartition_entities_by_instance_sequence,
    replace_components,
    suppress_hydrogens,
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


class TestPolymerSequence:
    def test_one_letter(self) -> None:
        sequence = PolymerSequence.from_one_letter("AG")
        assert sequence.to_one_letter() == "AG"
        assert tuple(component.component_id for component in sequence.components) == (
            "ALA",
            "GLY",
        )

    def test_nucleic_mapping(self) -> None:
        rna = PolymerSequence.from_one_letter("I", polymer_type="rna")
        dna = PolymerSequence.from_one_letter("IU", polymer_type="dna")

        assert [component.component_id for component in rna.components] == ["I"]
        assert [component.component_id for component in dna.components] == ["DI", "DU"]


class TestMolecule:
    def test_partial_chemistry(self) -> None:
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

    def test_from_smiles(self) -> None:
        molecule = Molecule.from_smiles("CCO")
        (component,) = molecule.components

        assert component.bonds is not None
        for atom in component.atoms:
            assert atom.formal_charge is not None
            assert atom.radical_electrons is not None
            assert atom.explicit_hydrogens is not None
            assert atom.implicit_hydrogens is not None
            assert isinstance(atom.implicit_hydrogens_allowed, bool)
            assert isinstance(atom.aromatic, bool)


class TestChemicalAtom:
    def test_defaults(self) -> None:
        atom = ChemicalAtom("N", "N")

        assert atom.atom_id == "N"
        assert atom.element == "N"
        assert atom.isotope is None
        assert atom.formal_charge is None
        assert atom.radical_electrons is None
        assert atom.explicit_hydrogens is None
        assert atom.implicit_hydrogens is None
        assert atom.implicit_hydrogens_allowed is None
        assert atom.aromatic is None
        assert atom.stereo is None

    def test_isotope(self) -> None:
        deuterium = ChemicalAtom("D1", "D")
        assert (deuterium.element, deuterium.isotope) == ("H", 2)


class TestChemicalComponent:
    def test_without_hydrogens(self) -> None:
        deuterium = ChemicalAtom("D1", "D")
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

    def test_unresolved_endpoint(self) -> None:
        with pytest.raises(ValueError, match="unknown atom"):
            ChemicalComponent(
                "BAD",
                (ChemicalAtom("C1", "C"),),
                frozenset({ChemicalBond("C1", "missing")}),
            )

    def test_partial_and_bondless(self) -> None:
        atoms = (ChemicalAtom("NA", "Na"),)

        partial = ChemicalComponent("X", atoms, None)
        bondless = ChemicalComponent("X", atoms, frozenset())

        assert partial.bonds is None
        assert bondless.bonds == frozenset()
        assert partial != bondless

    @pytest.mark.parametrize("bonds", [None, frozenset()])
    def test_unique_atom_ids(self, bonds: frozenset[ChemicalBond] | None) -> None:
        with pytest.raises(ValueError):
            ChemicalComponent("X", (ChemicalAtom("A", "C"), ChemicalAtom("A", "C")), bonds)

    @pytest.mark.parametrize(
        "bonds",
        [
            frozenset({ChemicalBond("A", "GHOST")}),
            frozenset({ChemicalBond("A", "A")}),
            frozenset({ChemicalBond("A", "B"), ChemicalBond("B", "A")}),
        ],
    )
    def test_bond_closure(self, bonds: frozenset[ChemicalBond]) -> None:
        atoms = (ChemicalAtom("A", "C"), ChemicalAtom("B", "C"))

        with pytest.raises(ValueError):
            ChemicalComponent("X", atoms, bonds)

    @pytest.mark.parametrize(
        "attachment",
        [PolymerAttachment("Z"), PolymerAttachment("A", frozenset({"A"}))],
    )
    def test_attachment_closure(self, attachment: PolymerAttachment) -> None:
        atoms = (ChemicalAtom("A", "C"), ChemicalAtom("B", "C"))

        with pytest.raises(ValueError):
            ChemicalComponent("X", atoms, None, previous_attachment=attachment)

    def test_leaving_atoms(self) -> None:
        atoms = (ChemicalAtom("N", "N"), ChemicalAtom("H", "H"))
        component = ChemicalComponent(
            "X",
            atoms,
            frozenset({ChemicalBond("N", "H")}),
            previous_attachment=PolymerAttachment("N", frozenset({"H"})),
            leaving_atom_ids=frozenset({"H"}),
            non_leaving_atom_ids=frozenset({"N"}),
        )

        assert component.without_hydrogens().leaving_atom_ids == frozenset()

    def test_reference_coordinates(self) -> None:
        atoms = (ChemicalAtom("C", "C"), ChemicalAtom("O", "O"))

        assert ChemicalComponent("LIG", atoms, frozenset()).reference_coordinates == ()
        aligned = ChemicalComponent(
            "LIG",
            atoms,
            frozenset(),
            reference_coordinates=((0.0, 0.0, 0.0), None),
        )
        assert len(aligned.reference_coordinates) == len(aligned.atoms)
        with pytest.raises(ValueError):
            ChemicalComponent("LIG", atoms, frozenset(), reference_coordinates=((0.0, 0.0, 0.0),))


class TestPolymerAttachment:
    def test_leaving_atoms(self) -> None:
        assert PolymerAttachment("N").leaving_atom_ids == frozenset()
        assert PolymerAttachment("N", frozenset({"H2"})).leaving_atom_ids == frozenset({"H2"})


class TestSuppressHydrogens:
    def test_realized_chemistry(self) -> None:
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("AHWT")},
            {"A": "protein"},
        )

        suppressed, old_to_new = suppress_hydrogens(system)

        assert not bool((suppressed.atomic_number == 1).any())
        assert old_to_new.shape == system.atom_symbol.shape
        removed = old_to_new < 0
        assert torch.equal(removed, system.atomic_number == 1)
        assert int(suppressed.explicit_hydrogens.sum()) == int(removed.sum())

    def test_alkene_stereo(self) -> None:
        def atom(atom_id: str, element: str) -> ChemicalAtom:
            return ChemicalAtom(
                atom_id,
                element,
                formal_charge=0,
                radical_electrons=0,
                explicit_hydrogens=0,
                implicit_hydrogens=0,
                implicit_hydrogens_allowed=False,
                aromatic=False,
            )

        component = ChemicalComponent(
            "ALKENE",
            tuple(
                atom(*values)
                for values in (("C1", "C"), ("C2", "C"), ("F", "F"), ("H", "H"), ("Cl", "Cl"), ("Br", "Br"))
            ),
            frozenset(
                {
                    ChemicalBond("C1", "C2", order=2, aromatic=False, stereo="E", stereo_atoms=("H", "Cl")),
                    ChemicalBond("C1", "F", aromatic=False),
                    ChemicalBond("C1", "H", aromatic=False),
                    ChemicalBond("C2", "Cl", aromatic=False),
                    ChemicalBond("C2", "Br", aromatic=False),
                }
            ),
        )
        system = MolecularSystem.from_entities({"alkene": Molecule((component,))}, {"A": "alkene"})

        suppressed, _ = suppress_hydrogens(system)
        double_bond = torch.nonzero(suppressed.bond_order == 2, as_tuple=False).flatten()

        assert double_bond.numel() == 1
        assert suppressed.symbols[int(suppressed.bond_stereo[double_bond[0]])] == "E"

    def test_hydride(self) -> None:
        component = ChemicalComponent(
            "HYDRIDE",
            (ChemicalAtom("C", "C"), ChemicalAtom("H", "H", formal_charge=-1)),
            frozenset({ChemicalBond("C", "H")}),
        )
        system = MolecularSystem.from_entities({"hydride": Molecule((component,))}, {"A": "hydride"})

        with pytest.raises(ValueError, match="charged or radical hydrogen"):
            suppress_hydrogens(system)

    @pytest.mark.parametrize(
        "hydrogen",
        (
            ChemicalAtom("H", "H", formal_charge=1),
            ChemicalAtom("H", "H", radical_electrons=1),
            ChemicalAtom("H", "H", formal_charge=0),
            ChemicalAtom("H", "H", radical_electrons=0),
        ),
    )
    def test_disconnected_state(
        self,
        hydrogen: ChemicalAtom,
    ) -> None:
        component = ChemicalComponent("HYDROGEN", (hydrogen,), frozenset())
        system = MolecularSystem.from_entities({"hydrogen": Molecule((component,))}, {"A": "hydrogen"})

        with pytest.raises(ValueError, match="cannot suppress"):
            suppress_hydrogens(system)

    def test_disconnected_neutral(self) -> None:
        component = ChemicalComponent(
            "HYDROGEN",
            (
                ChemicalAtom("H1", "H", formal_charge=0, radical_electrons=0),
                ChemicalAtom("H2", "H", formal_charge=0, radical_electrons=0),
            ),
            frozenset({ChemicalBond("H1", "H2")}),
        )
        system = MolecularSystem.from_entities({"hydrogen": Molecule((component,))}, {"A": "hydrogen"})

        suppressed, old_to_new = suppress_hydrogens(system)

        assert suppressed.atom_symbol.numel() == 0
        assert old_to_new.tolist() == [-1, -1]


class TestMolecularSystem:
    def test_materialization(self) -> None:
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

    def test_duplicate_edge(self) -> None:
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

    def test_instance_values(self) -> None:
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


def test_water_component_mask_category() -> None:
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


def test_instance_subgraph_selection() -> None:
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


def test_molecular_system_batching() -> None:
    system = MolecularSystem.from_entities(
        {"protein": PolymerSequence("protein", (_residue("R0"),))},
        {"B": "protein"},
    )

    batched = MolecularSystem.batch((system, system))
    assert batched.system_entity_ptr.tolist() == [0, 1, 2]
    assert batched.system_instance_ptr.tolist() == [0, 1, 2]
    first, second = batched.unbatch()
    assert first.instance_id(0) == second.instance_id(0) == "B"
    assert first.atom_symbol.tolist() == second.atom_symbol.tolist()


def test_replace_components_polymer_junction() -> None:
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


def test_repartition_entities() -> None:
    original = MolecularSystem.from_entities(
        {"protein": PolymerSequence("protein", (_residue("R0"), _residue("R1")))},
        {"A": "protein", "B": "protein"},
    )
    realized, _, _ = replace_components(
        original,
        torch.tensor([0], dtype=torch.int64),
        (_residue("R2"),),
    )

    repartitioned = repartition_entities_by_instance_sequence(original, realized)

    assert [repartitioned.entity_id(index) for index in range(2)] == ["protein", "protein_2"]
    assert repartitioned.instance_entity.tolist() == [1, 0]


def test_form_bond_hydrogen_counts() -> None:
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


class TestCopyComponents:
    def test_repeated_source(self) -> None:
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

    def test_stereo(self) -> None:
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


def test_disjoint_union_offsets() -> None:
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


def test_component_boundaries() -> None:
    sequence = PolymerSequence("protein", (_residue("A"), _residue("B")))
    system = MolecularSystem.from_entities({"protein": sequence}, {"A": "protein"})
    first = system.atom_index(AtomSite("A", 0, "C"))
    second = system.atom_index(AtomSite("A", 1, "N"))

    assert component_boundary_atoms(system) == ((first,), (second,))

    atom_mask = torch.ones(system.atom_symbol.shape, dtype=torch.bool)
    atom_mask[second] = False
    assert component_boundary_atoms(system, atom_mask) == ((), ())


def test_symbol_replacement():
    system = MolecularSystem.from_entities({"query": PolymerSequence.from_one_letter("A")}, {"A": "query"})
    symbols = list(system.symbols)
    restored = system.replace(symbols=symbols)
    symbols[0] = "changed"
    assert restored.symbols == system.symbols
    assert isinstance(restored.symbols, tuple)
