from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ofoldx.data import (
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    MOLECULE_BRANCHED,
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerAttachment,
    PolymerSequence,
    Structure,
    SystemBond,
    encode_system,
    get_component,
)
from ofoldx.io import read_structure, write_structure
from ofoldx.io.mmcif import (
    _cif_bond_order,
    _cif_component_bond_order,
    _read_mmcif_entity_declarations,
    _remove_component_atoms,
    read_mmcif_components,
    read_mmcif_branch_entities,
)
from ofoldx.io.structure import NoRetainedStructureError, _resolve_branch_component_index


def _ligand_component(
    *,
    bond_order: float | None = 1.0,
    bond_aromatic: bool | None = False,
) -> ChemicalComponent:
    return ChemicalComponent(
        "LIG",
        (
            ChemicalAtom("C1", "C", formal_charge=0),
            ChemicalAtom("O1", "O", formal_charge=-1),
        ),
        frozenset({ChemicalBond("C1", "O1", order=bond_order, aromatic=bond_aromatic)}),
        reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
    )


def _water_component() -> ChemicalComponent:
    return ChemicalComponent(
        "HOH",
        (
            ChemicalAtom("O", "O"),
            ChemicalAtom("H1", "H"),
            ChemicalAtom("H2", "H"),
        ),
        frozenset({ChemicalBond("O", "H1"), ChemicalBond("O", "H2")}),
        reference_coordinates=((0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)),
    )


def _structure(
    *,
    bond_order: float | None = 1.0,
    bond_aromatic: bool | None = False,
) -> Structure:
    component = _ligand_component(bond_order=bond_order, bond_aromatic=bond_aromatic)
    system = MolecularSystem.from_entities(
        {"ligand": Molecule((component,))},
        {"L": "ligand"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor([[1.0, 2.0, 3.0], [2.2, 2.0, 3.0]]),
        coordinate_mask=torch.ones(2, dtype=torch.bool),
        occupancy=torch.tensor([1.0, 0.5]),
        occupancy_mask=torch.ones(2, dtype=torch.bool),
        b_factor=torch.tensor([10.0, 20.0]),
        b_factor_mask=torch.ones(2, dtype=torch.bool),
        source_symbols=("11",),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([-1]),
        batch_size=[],
    )


def _read_cif(path: Path, text: str) -> object:
    from biotite.structure.io import pdbx

    path.write_text(text)
    return pdbx.CIFFile.read(str(path))


def test_remove_component_atoms_without_topology_removes_only_declared_atoms() -> None:
    component = ChemicalComponent(
        "LEFT",
        (
            ChemicalAtom("A", "C"),
            ChemicalAtom("L", "O"),
            ChemicalAtom("H", "H"),
        ),
        None,
        previous_attachment=PolymerAttachment("A", frozenset({"L"})),
        next_attachment=PolymerAttachment("L"),
        reference_coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )

    retained = _remove_component_atoms(component, {"L"})

    assert tuple(atom.atom_id for atom in retained.atoms) == ("A", "H")
    assert retained.reference_coordinates == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert retained.bonds is None
    assert retained.previous_attachment == PolymerAttachment("A")
    assert retained.next_attachment is None


def test_remove_component_atoms_with_known_topology_still_expands_neighbor_hydrogen() -> None:
    component = ChemicalComponent(
        "LEFT",
        (
            ChemicalAtom("A", "C"),
            ChemicalAtom("L", "O"),
            ChemicalAtom("H", "H"),
        ),
        frozenset({ChemicalBond("A", "L"), ChemicalBond("L", "H")}),
    )

    retained = _remove_component_atoms(component, {"L"})

    assert tuple(atom.atom_id for atom in retained.atoms) == ("A",)
    assert retained.bonds == frozenset()


def test_branch_link_survives_leaving_atom_removal_without_component_topology(tmp_path: Path) -> None:
    cif_file = _read_cif(
        tmp_path / "branch-no-topology.cif",
        """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 LEFT 1
1 RIGHT 2
loop_
_pdbx_entity_branch_link.entity_id
_pdbx_entity_branch_link.entity_branch_list_num_1
_pdbx_entity_branch_link.comp_id_1
_pdbx_entity_branch_link.atom_id_1
_pdbx_entity_branch_link.leaving_atom_id_1
_pdbx_entity_branch_link.entity_branch_list_num_2
_pdbx_entity_branch_link.comp_id_2
_pdbx_entity_branch_link.atom_id_2
_pdbx_entity_branch_link.leaving_atom_id_2
_pdbx_entity_branch_link.value_order
1 1 LEFT A L 2 RIGHT B ? sing
""",
    )
    left = ChemicalComponent(
        "LEFT",
        (ChemicalAtom("A", "C"), ChemicalAtom("L", "O")),
        None,
    )
    right = ChemicalComponent("RIGHT", (ChemicalAtom("B", "N"),), None)

    entities, _ = read_mmcif_branch_entities(cif_file, {"LEFT": left, "RIGHT": right})
    branch = entities["1"]

    assert tuple(atom.atom_id for atom in branch.components[0].atoms) == ("A",)
    assert branch.components[0].bonds is None
    assert len(branch.bonds) == 1


def _fully_observed_structure(system: MolecularSystem) -> Structure:
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros((atom_count, 3)),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.ones(atom_count),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=tuple(str(index + 1) for index in range(component_count)),
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1),
        batch_size=[],
    )


def _bonded_molecules() -> tuple[Structure, dict[str, ChemicalComponent]]:
    left = ChemicalComponent("LEFT", (ChemicalAtom("L", "C"),), frozenset())
    right = ChemicalComponent("RIGHT", (ChemicalAtom("R", "N"),), frozenset())
    system = MolecularSystem.from_entities(
        {"left": Molecule((left,)), "right": Molecule((right,))},
        {"A": "left", "B": "right"},
        (SystemBond(AtomSite("A", 0, "L"), AtomSite("B", 0, "R"), order=2.0),),
    )
    return (
        _fully_observed_structure(system),
        {"LEFT": left, "RIGHT": right},
    )


class TestMmcifIO:
    def test_water_only_defaults_to_excluded_and_can_be_opted_in(self, tmp_path: Path) -> None:
        water = _water_component()
        system = MolecularSystem.from_entities(
            {"water": Molecule((water,), molecule_type="water")},
            {"W": "water"},
        )
        path = tmp_path / "water-only.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="only water excluded"):
            read_structure(path, components={"HOH": water})
        retained = read_structure(path, components={"HOH": water}, include_water=True)

        assert retained.system.atomic_number.tolist() == [8]
        assert retained.system.instance_id(0) == "W"

    def test_mixed_water_defaults_to_only_the_nonwater_entity(self, tmp_path: Path) -> None:
        ligand = _ligand_component()
        water = _water_component()
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "water": Molecule((water,), molecule_type="water"),
            },
            {"L": "ligand", "W": "water"},
        )
        path = tmp_path / "mixed-water.cif"
        write_structure(_fully_observed_structure(system), path)

        retained = read_structure(path, components={"HOH": water})

        assert retained.system.instance_symbol.numel() == 1
        assert retained.system.instance_id(0) == "L"
        assert {
            retained.system.symbols[int(symbol)]
            for symbol in retained.system.component_symbol.tolist()
        } == {"LIG"}

    def test_unobserved_declared_water_asym_is_not_rematerialized_by_default(self, tmp_path: Path) -> None:
        ligand = _ligand_component()
        water = _water_component()
        system = MolecularSystem.from_entities(
            {
                "ligand": Molecule((ligand,)),
                "water": Molecule((water,), molecule_type="water"),
            },
            {"L": "ligand", "W": "water"},
        )
        structure = _fully_observed_structure(system)
        water_start = int(system.component_atom_ptr[1])
        coordinate_mask = structure.coordinate_mask.clone()
        occupancy_mask = structure.occupancy_mask.clone()
        b_factor_mask = structure.b_factor_mask.clone()
        coordinate_mask[water_start:] = False
        occupancy_mask[water_start:] = False
        b_factor_mask[water_start:] = False
        structure = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=coordinate_mask,
            occupancy=structure.occupancy.clone(),
            occupancy_mask=occupancy_mask,
            b_factor=structure.b_factor.clone(),
            b_factor_mask=b_factor_mask,
        )
        path = tmp_path / "declared-water.cif"
        write_structure(structure, path)

        default = read_structure(path, components={"HOH": water})
        opted_in = read_structure(path, components={"HOH": water}, include_water=True)

        assert default.system.instance_symbol.numel() == 1
        assert default.system.instance_id(0) == "L"
        assert opted_in.system.instance_symbol.numel() == 2
        water_instance = next(
            index
            for index in range(int(opted_in.system.instance_symbol.numel()))
            if opted_in.system.instance_id(index) == "W"
        )
        water_component = int(opted_in.system.instance_component_ptr[water_instance])
        water_atom = int(opted_in.system.component_atom_ptr[water_component])
        assert int(opted_in.system.atomic_number[water_atom]) == 8
        assert not bool(opted_in.coordinate_mask[water_atom])

    def test_struct_conn_whitelist_forces_single_undirected_inter_component_bond(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / "covalent.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"]["ptnr1_role"] = pdbx.CIFColumn(["metal"])
        cif_file.block["struct_conn"]["ptnr2_role"] = pdbx.CIFColumn(["donor"])
        cif_file.write(str(path))

        restored = read_structure(path, components=components)

        assert restored.system.edge_index.shape[1] == 1
        assert restored.system.bond_order.tolist() == [1.0]
        assert restored.system.bond_kind.tolist() == [BOND_KIND_COVALENT]
        assert restored.system.bond_direction.tolist() == [BOND_DIRECTION_UNDIRECTED]

    @pytest.mark.parametrize("connection_type", ["metalc", "covalent", "hydrog"])
    def test_struct_conn_ignores_unsupported_connection_types(
        self,
        tmp_path: Path,
        connection_type: str,
    ) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / f"unsupported-{connection_type}.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["struct_conn"]["conn_type_id"] = pdbx.CIFColumn([connection_type])
        cif_file.write(str(path))

        assert read_structure(path, components=components).system.edge_index.shape[1] == 0

    def test_struct_conn_ignores_same_component_links(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        structure, components = _bonded_molecules()
        path = tmp_path / "same-component.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        category = cif_file.block["struct_conn"]
        for suffix in (
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
            "auth_asym_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_atom_id",
        ):
            category[f"ptnr2_{suffix}"] = pdbx.CIFColumn(category[f"ptnr1_{suffix}"].as_array())
        cif_file.write(str(path))

        assert read_structure(path, components=components).system.edge_index.shape[1] == 0

    def test_asx_rows_and_declarations_normalize_to_asp(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("D")},
            {"A": "protein"},
        )
        path = tmp_path / "asx.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        for column_name in ("label_comp_id", "auth_comp_id"):
            atom_site[column_name] = pdbx.CIFColumn(["ASX"] * atom_site.row_count)
        for column_name in ("label_atom_id", "auth_atom_id"):
            atom_site[column_name] = pdbx.CIFColumn(
                [
                    {"OD1": "XD1", "OD2": "XD2"}.get(atom_id, atom_id)
                    for atom_id in atom_site[column_name].as_array()
                ]
            )
        atom_site["type_symbol"] = pdbx.CIFColumn(
            [
                "X" if atom_id in {"XD1", "XD2"} else element
                for atom_id, element in zip(
                    atom_site["label_atom_id"].as_array(),
                    atom_site["type_symbol"].as_array(),
                )
            ]
        )
        cif_file.block["entity_poly_seq"]["mon_id"] = pdbx.CIFColumn(["ASX"])
        cif_file.write(str(path))

        restored = read_structure(path)
        component_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.component_symbol.tolist()
        }
        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert component_ids == {"ASP"}
        assert {"OD1", "OD2"} <= atom_ids
        assert not {"XD1", "XD2"}.intersection(atom_ids)

    def test_raw_unl_occurrence_is_removed_without_explicit_custom_component(self, tmp_path: Path) -> None:
        unl = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("A"), "unl": Molecule((unl,))},
            {"A": "protein", "L": "unl"},
        )
        path = tmp_path / "raw-unl.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)

        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]
        restored_custom = read_structure(path, components={"UNL": unl})
        assert {
            restored_custom.system.instance_id(index)
            for index in range(restored_custom.system.instance_symbol.numel())
        } == {"A", "L"}

    @pytest.mark.parametrize("component_id", ["UNX", "UNL"])
    def test_only_raw_excluded_occurrences_raise_no_retained_structure(
        self,
        tmp_path: Path,
        component_id: str,
    ) -> None:
        component = ChemicalComponent(component_id, (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities({"excluded": Molecule((component,))}, {"L": "excluded"})
        path = tmp_path / f"raw-{component_id.lower()}.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="excluded UNX/UNL"):
            read_structure(path)

    def test_raw_unl_branched_declaration_is_skipped_as_no_retained_structure(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities({"excluded": Molecule((component,))}, {"L": "excluded"})
        system = system.replace(entity_molecule_type=system.entity_molecule_type.new_tensor([MOLECULE_BRANCHED]))
        path = tmp_path / "raw-unl-branched.cif"
        write_structure(_fully_observed_structure(system), path)

        with pytest.raises(NoRetainedStructureError, match="excluded UNX/UNL"):
            read_structure(path)

    def test_raw_unl_branched_declaration_does_not_break_retained_entities(self, tmp_path: Path) -> None:
        component = ChemicalComponent("UNL", (ChemicalAtom("X", "C"),), frozenset())
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("A"), "excluded": Molecule((component,))},
            {"A": "protein", "L": "excluded"},
        )
        molecule_types = system.entity_molecule_type.clone()
        molecule_types[1] = MOLECULE_BRANCHED
        system = system.replace(entity_molecule_type=molecule_types)
        path = tmp_path / "mixed-raw-unl-branched.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)
        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]

    def test_declared_mixed_protein_is_retained_when_only_unk_is_observed(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        sequence = PolymerSequence("protein", (get_component("UNK"), get_component("ALA")))
        system = MolecularSystem.from_entities({"protein": sequence}, {"A": "protein"})
        path = tmp_path / "declared-mixed-observed-unk.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        keep = [value == "1" for value in atom_site["label_seq_id"].as_array()]
        cif_file.block["atom_site"] = pdbx.CIFCategory(
            {
                name: pdbx.CIFColumn([value for value, retained in zip(atom_site[name].as_array(), keep) if retained])
                for name in atom_site
            }
        )
        cif_file.write(str(path))

        restored = read_structure(path)
        assert [restored.system.instance_id(index) for index in range(restored.system.instance_symbol.numel())] == [
            "A"
        ]
        assert [
            restored.system.symbols[int(symbol)] for symbol in restored.system.component_symbol.tolist()
        ] == ["UNK", "ALA"]
        atom_start = int(restored.system.component_atom_ptr[1])
        atom_stop = int(restored.system.component_atom_ptr[2])
        assert not bool(restored.coordinate_mask[atom_start:atom_stop].any())

    @pytest.mark.parametrize(("polymer_type", "component_id"), [("rna", "N"), ("dna", "DN")])
    def test_unknown_nucleic_polymer_is_not_removed(
        self,
        tmp_path: Path,
        polymer_type: str,
        component_id: str,
    ) -> None:
        sequence = PolymerSequence(polymer_type, (get_component(component_id),))
        system = MolecularSystem.from_entities({"polymer": sequence}, {"A": "polymer"})
        path = tmp_path / f"unknown-{component_id.lower()}.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)
        assert restored.system.instance_symbol.numel() == 1
        assert restored.system.symbols[int(restored.system.component_symbol[0])] == component_id

    def test_round_trip_preserves_the_complete_public_graph_and_coordinates(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        path = tmp_path / "ligand.cif"
        write_structure(structure, path)
        restored = read_structure(path, components={"LIG": _ligand_component()})

        expected_metadata, expected_arrays = encode_system(structure.system)
        actual_metadata, actual_arrays = encode_system(restored.system)
        assert actual_metadata == expected_metadata
        for name, expected in expected_arrays.items():
            assert torch.equal(actual_arrays[name], expected)
        assert torch.equal(restored.coordinates, structure.coordinates)
        assert torch.equal(restored.coordinate_mask, structure.coordinate_mask)
        assert torch.equal(restored.occupancy, structure.occupancy)
        assert restored.source_symbols == structure.source_symbols

    def test_atom_site_is_the_coordinate_source_of_truth(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "coordinates.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        values = cif_file.block["atom_site"]["Cartn_x"].as_array()
        values[0] = 123.0
        cif_file.block["atom_site"]["Cartn_x"] = pdbx.CIFColumn(values)
        cif_file.write(str(path))

        assert read_structure(path).coordinates[0, 0].item() == 123.0

    def test_ccd_miss_keeps_observed_atoms_without_embedded_chemistry(self, tmp_path: Path) -> None:
        path = tmp_path / "ccd-miss.cif"
        write_structure(_structure(), path)

        restored = read_structure(path)

        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert atom_ids == {"C1", "O1"}
        assert restored.system.edge_index.shape[1] == 0
        assert not restored.system.reference_coordinate_mask.any()

    def test_occurrence_charge_and_isotope_are_not_shared_through_component_placeholders(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        component = ChemicalComponent("DUP", (ChemicalAtom("H1", "H"),), frozenset())
        system = MolecularSystem.from_entities(
            {"ligand": Molecule((component,))},
            {"A": "ligand", "B": "ligand"},
        )
        path = tmp_path / "occurrence-fields.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        atom_site["type_symbol"] = pdbx.CIFColumn(["D", "T"])
        atom_site["pdbx_formal_charge"] = pdbx.CIFColumn(["-1", "1"])
        cif_file.write(str(path))

        for restored in (
            read_structure(path, include_hydrogens=True),
            read_structure(path, components={"DUP": component}, include_hydrogens=True),
        ):
            first = restored.system.atom_index(AtomSite("A", 0, "H1"))
            second = restored.system.atom_index(AtomSite("B", 0, "H1"))
            assert restored.system.isotope_known[[first, second]].tolist() == [True, True]
            assert restored.system.isotope[[first, second]].tolist() == [2, 3]
            assert restored.system.formal_charge_known[[first, second]].tolist() == [True, True]
            assert restored.system.formal_charge[[first, second]].tolist() == [-1, 1]

    def test_ccd_miss_occurrence_values_and_known_masks_do_not_leak_between_instances(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        component = ChemicalComponent("DUP", (ChemicalAtom("H1", "H"),), frozenset())
        system = MolecularSystem.from_entities(
            {"ligand": Molecule((component,))},
            {"A": "ligand", "B": "ligand"},
        )
        path = tmp_path / "occurrence-known-masks.cif"
        write_structure(_fully_observed_structure(system), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        atom_site["type_symbol"] = pdbx.CIFColumn(["D", "H"])
        atom_site["pdbx_formal_charge"] = pdbx.CIFColumn(["-1", "?"])
        cif_file.write(str(path))

        restored = read_structure(path, include_hydrogens=True)
        first = restored.system.atom_index(AtomSite("A", 0, "H1"))
        second = restored.system.atom_index(AtomSite("B", 0, "H1"))
        assert bool(restored.system.isotope_known[first])
        assert int(restored.system.isotope[first]) == 2
        assert not bool(restored.system.isotope_known[second])
        assert int(restored.system.isotope[second]) == 0
        assert bool(restored.system.formal_charge_known[first])
        assert int(restored.system.formal_charge[first]) == -1
        assert not bool(restored.system.formal_charge_known[second])
        assert int(restored.system.formal_charge[second]) == 0

    def test_known_component_drops_extra_atom_site_rows(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "known-extra.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_site = cif_file.block["atom_site"]
        columns = {}
        for name in atom_site:
            values = atom_site[name].as_array().tolist()
            extra = values[0]
            if name == "id":
                extra = "3"
            elif name in {"label_atom_id", "auth_atom_id"}:
                extra = "H1"
            elif name == "type_symbol":
                extra = "H"
            elif name == "pdbx_formal_charge":
                extra = "?"
            values.append(extra)
            columns[name] = pdbx.CIFColumn(values)
        cif_file.block["atom_site"] = pdbx.CIFCategory(columns)
        cif_file.write(str(path))

        restored = read_structure(path, components={"LIG": _ligand_component()})
        atom_ids = {
            restored.system.symbols[int(symbol)]
            for symbol in restored.system.atom_symbol.tolist()
        }
        assert atom_ids == {"C1", "O1"}

    def test_nonpoly_atom_without_coordinates_is_absent_from_the_observed_union(
        self,
        tmp_path: Path,
    ) -> None:
        structure = _structure()
        masked = structure.replace(
            coordinates=structure.coordinates.clone(),
            coordinate_mask=torch.tensor([False, True]),
            occupancy=structure.occupancy.clone(),
            occupancy_mask=torch.tensor([False, True]),
            b_factor=structure.b_factor.clone(),
            b_factor_mask=torch.tensor([False, True]),
        )
        masked.coordinates[0] = torch.nan
        path = tmp_path / "masked.cif"

        write_structure(masked, path)
        restored = read_structure(path, components={"LIG": _ligand_component()})

        assert restored.system.atom_symbol.numel() == 1
        assert restored.system.symbols[int(restored.system.atom_symbol[0])] == "O1"
        assert restored.coordinate_mask.tolist() == [True]

    def test_aromaticity_masks_are_not_replaced_by_filler_values(
        self,
        tmp_path: Path,
    ) -> None:
        partial_bond = _structure(bond_order=1.0, bond_aromatic=None)
        partial_path = tmp_path / "partial-bond.cif"
        write_structure(partial_bond, partial_path)
        restored = read_structure(
            partial_path,
            components={"LIG": _ligand_component(bond_order=1.0, bond_aromatic=None)},
        )
        assert restored.system.bond_order_known.all()
        assert not restored.system.bond_aromatic_known.any()

    @pytest.mark.parametrize("order", [1.0, 2.0])
    def test_component_bond_known_order_and_aromatic_true_round_trip(
        self,
        tmp_path: Path,
        order: float,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / f"aromatic-{order}.cif"
        write_structure(_structure(bond_order=order, bond_aromatic=True), path)

        cif_file = pdbx.CIFFile.read(str(path))
        bond_category = cif_file.block["chem_comp_bond"]
        assert bond_category["value_order"].as_array().tolist() == ["sing" if order == 1.0 else "doub"]
        assert bond_category["pdbx_aromatic_flag"].as_array().tolist() == ["Y"]

        restored = read_structure(
            path,
            components={"LIG": _ligand_component(bond_order=order, bond_aromatic=True)},
        )
        assert restored.system.bond_order_known.all()
        assert restored.system.bond_order.tolist() == [order]
        assert restored.system.bond_aromatic_known.all()
        assert restored.system.bond_aromatic.all()

    def test_component_bond_unknown_order_and_aromaticity_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown-bond.cif"
        write_structure(_structure(bond_order=None, bond_aromatic=None), path)
        restored = read_structure(
            path,
            components={"LIG": _ligand_component(bond_order=None, bond_aromatic=None)},
        )

        assert not restored.system.bond_order_known.any()
        assert not restored.system.bond_aromatic_known.any()

    @pytest.mark.parametrize(
        ("order", "aromatic"),
        [
            ("arom", "N"),
            ("sing", "Y"),
            ("sing", "invalid"),
        ],
    )
    def test_reader_ignores_inconsistent_or_invalid_embedded_aromatic_flags(
        self,
        tmp_path: Path,
        order: str,
        aromatic: str,
    ) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "invalid-aromatic.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        bond_category = cif_file.block["chem_comp_bond"]
        bond_category["value_order"] = pdbx.CIFColumn([order])
        bond_category["pdbx_aromatic_flag"] = pdbx.CIFColumn([aromatic])
        cif_file.write(str(path))

        components, _ = read_mmcif_components(pdbx.CIFFile.read(str(path)))
        bond = next(iter(components["LIG"].bonds or ()))
        assert bond.order == (1.5 if order == "arom" else 1.0)
        assert bond.aromatic == ({"Y": True, "N": False}.get(aromatic))

    def test_reader_maps_invalid_embedded_atom_aromatic_flag_to_unknown(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "invalid-atom-aromatic.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))
        atom_category = cif_file.block["chem_comp_atom"]
        atom_category["pdbx_aromatic_flag"] = pdbx.CIFColumn(["invalid"] * atom_category.row_count)
        cif_file.write(str(path))

        components, _ = read_mmcif_components(pdbx.CIFFile.read(str(path)))
        assert all(atom.aromatic is None for atom in components["LIG"].atoms)

    def test_default_reader_omits_component_edges_but_keeps_polymer_adjacency(
        self,
        tmp_path: Path,
    ) -> None:
        system = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter("AA")},
            {"A": "protein"},
        )
        path = tmp_path / "polymer.cif"
        write_structure(_fully_observed_structure(system), path)

        restored = read_structure(path)

        assert not restored.system.component_topology_complete.any()
        assert not bool((restored.system.bond_role == BOND_ROLE_COMPONENT).any())
        assert (restored.system.bond_role == BOND_ROLE_POLYMER).sum().item() == 1

    def test_explicit_components_preserve_component_topology(self, tmp_path: Path) -> None:
        path = tmp_path / "explicit.cif"
        write_structure(_structure(), path)

        restored = read_structure(path, components={"LIG": _ligand_component()})

        assert restored.system.component_topology_complete.tolist() == [True]
        assert (restored.system.bond_role == BOND_ROLE_COMPONENT).sum().item() == 1

    def test_struct_conn_bond_order_semantics_are_unchanged(self) -> None:
        assert _cif_bond_order("sing") == (1.0, False)
        assert _cif_bond_order("doub") == (2.0, False)
        assert _cif_component_bond_order("sing") == (1.0, None)
        assert _cif_component_bond_order("doub") == (2.0, None)

    def test_nonpoly_numbering_uses_ndb_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "numbering.cif",
            """data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
_pdbx_nonpoly_scheme.pdb_seq_num
_pdbx_nonpoly_scheme.pdb_ins_code
A 1 10 10 .
A 2 20 20 ?
""",
        )

        numbering = _read_mmcif_entity_declarations(cif_file)[3]
        assert numbering == {("A", 1): ("10", None), ("A", 2): ("20", None)}

    @pytest.mark.parametrize(
        ("ndb_number", "message"),
        [("?", "missing asym_id or ndb_seq_num"), ("not-an-integer", "expected an integer identity")],
    )
    def test_nonpoly_numbering_requires_integer_ndb_sequence_number(
        self,
        tmp_path: Path,
        ndb_number: str,
        message: str,
    ) -> None:
        cif_file = _read_cif(
            tmp_path / "invalid-numbering.cif",
            f"""data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
A {ndb_number} 10
""",
        )

        with pytest.raises(ValueError, match=message):
            _read_mmcif_entity_declarations(cif_file)

    def test_nonpoly_numbering_rejects_a_true_key_conflict(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "conflicting-numbering.cif",
            """data_numbering
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
A 1 10
A 1 20
""",
        )

        with pytest.raises(ValueError, match="conflicting numbering for \\('A', 1\\)"):
            _read_mmcif_entity_declarations(cif_file)

    def test_read_structure_accepts_multiple_nonpoly_numbers_for_one_asym_id(self, tmp_path: Path) -> None:
        from biotite.structure.io import pdbx

        path = tmp_path / "multiple-nonpoly.cif"
        write_structure(_structure(), path)
        cif_file = pdbx.CIFFile.read(str(path))

        scheme = cif_file.block["pdbx_nonpoly_scheme"]
        expanded_scheme = {}
        for column_name in scheme:
            values = scheme[column_name].as_array().tolist()
            if column_name == "ndb_seq_num":
                values.append("2")
            elif column_name in {"auth_seq_num", "pdb_seq_num"}:
                values.append("20")
            else:
                values.append(values[0])
            expanded_scheme[column_name] = pdbx.CIFColumn(values)
        cif_file.block["pdbx_nonpoly_scheme"] = pdbx.CIFCategory(expanded_scheme)

        atom_site = cif_file.block["atom_site"]
        original_row_count = atom_site.row_count
        expanded_atom_site = {}
        for column_name in atom_site:
            values = atom_site[column_name].as_array().tolist()
            extra = list(values)
            if column_name == "id":
                extra = [str(original_row_count + index + 1) for index in range(original_row_count)]
            elif column_name == "auth_seq_id":
                extra = ["20"] * original_row_count
            values.extend(extra)
            expanded_atom_site[column_name] = pdbx.CIFColumn(values)
        cif_file.block["atom_site"] = pdbx.CIFCategory(expanded_atom_site)
        cif_file.write(str(path))

        restored = read_structure(path)
        assert restored.system.instance_entity.numel() == 2

    def test_branch_definition_cannot_override_the_entity_category(
        self,
        tmp_path: Path,
    ) -> None:
        from biotite.structure.io import pdbx

        structure = _structure()
        structure = structure.replace(
            system=structure.system.replace(
                entity_molecule_type=structure.system.entity_molecule_type.new_tensor([MOLECULE_BRANCHED])
            )
        )
        path = tmp_path / "conflicting-category.cif"
        write_structure(structure, path)
        cif_file = pdbx.CIFFile.read(str(path))
        cif_file.block["entity"]["type"] = pdbx.CIFColumn(["non-polymer"])
        cif_file.write(str(path))

        with pytest.raises(ValueError, match="branched-entity definition"):
            read_structure(path)

    def test_branch_scheme_keeps_label_pdb_and_author_namespaces_paired(self, tmp_path: Path) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-namespaces.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
1 X 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
B 1 X 1 P PX 10 A AX 20
B 1 X 2 Q QX 20 C CX 30
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"X": component})

        assert candidates[("pdb", "B", "1", "PX", "P", "10")] == frozenset({0})
        assert candidates[("author", "B", "1", "AX", "A", "20")] == frozenset({0})
        assert ("pdb", "B", "1", "PX", "A", "20") not in candidates
        assert candidates[("label", "B", "1", "X", "B", None)] == frozenset({0, 1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="1",
            component_id="X",
            auth_component_id="PX",
            auth_chain_id="P",
            auth_residue_id="10",
        )
        assert _resolve_branch_component_index(row, candidates, ("B", 10)) == 0

    def test_atom_site_auth_identity_uses_branch_pdb_namespace(self, tmp_path: Path) -> None:
        component = ChemicalComponent("NAG", (ChemicalAtom("C1", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-auth-collision.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
2 NAG 1
2 NAG 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
B 2 NAG 1 B NAG 1 B NAG 0
B 2 NAG 2 B NAG 2 B NAG 1
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"NAG": component})
        assert candidates[("label", "B", "2", "NAG", "B", None)] == frozenset({0, 1})
        assert candidates[("pdb", "B", "2", "NAG", "B", "1")] == frozenset({0})
        assert candidates[("author", "B", "2", "NAG", "B", "1")] == frozenset({1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="2",
            component_id="NAG",
            auth_component_id="NAG",
            auth_chain_id="B",
            auth_residue_id="1",
        )
        assert _resolve_branch_component_index(row, candidates, (None, "B", 1, "1", None, "NAG")) == 0

    def test_branch_author_namespace_does_not_disambiguate_atom_site_auth(self) -> None:
        candidates = {
            ("label", "B", "2", "NAG", "B", None): frozenset({0, 1}),
            ("author", "B", "2", "NAG", "B", "1"): frozenset({0}),
        }
        row = SimpleNamespace(
            chain_id="B",
            entity_id="2",
            component_id="NAG",
            auth_component_id="NAG",
            auth_chain_id="B",
            auth_residue_id="1",
        )
        with pytest.raises(ValueError, match="maps ambiguously"):
            _resolve_branch_component_index(row, candidates, (None, "B", 1, "1", None, "NAG"))

    def test_branch_source_numbering_uses_pdb_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "branch-numbering.cif",
            """data_branch
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 10 20 ?
""",
        )
        _, _, _, author_numbering, _, _ = _read_mmcif_entity_declarations(cif_file)
        assert author_numbering[("B", 1)] == ("10", None)

    def test_branch_source_numbering_does_not_fallback_to_auth_sequence_number(self, tmp_path: Path) -> None:
        cif_file = _read_cif(
            tmp_path / "branch-numbering-no-fallback.cif",
            """data_branch
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 ? 20 ?
""",
        )
        _, _, _, author_numbering, _, _ = _read_mmcif_entity_declarations(cif_file)
        assert author_numbering[("B", 1)] == (None, None)

    @pytest.mark.parametrize("observe_second_nag", [False, True])
    def test_repeated_branch_component_struct_conn_uses_pdb_identity(
        self,
        tmp_path: Path,
        observe_second_nag: bool,
    ) -> None:
        second_nag = (
            "HETATM 2 C C1 . NAG B 1 . ? 2.0 0.0 0.0 1.0 10.0 ? 2 NAG B C1 1\n"
            if observe_second_nag
            else ""
        )
        ligand_serial = 3 if observe_second_nag else 2
        path = tmp_path / "repeated-branch-struct-conn.cif"
        path.write_text(
            """data_branch
loop_
_entity.id
_entity.type
1 branched
2 non-polymer
loop_
_struct_asym.id
_struct_asym.entity_id
B 1
L 2
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 NAG 1
1 NAG 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_mon_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_asym_id
_pdbx_branch_scheme.auth_mon_id
_pdbx_branch_scheme.auth_seq_num
_pdbx_branch_scheme.pdb_ins_code
B 1 NAG 1 B NAG 1 B NAG 0 ?
B 1 NAG 2 B NAG 2 B NAG 1 ?
loop_
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.comp_id
2 LIG
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.entity_id
_pdbx_nonpoly_scheme.mon_id
_pdbx_nonpoly_scheme.ndb_seq_num
_pdbx_nonpoly_scheme.pdb_seq_num
_pdbx_nonpoly_scheme.auth_seq_num
_pdbx_nonpoly_scheme.pdb_ins_code
L 2 LIG 1 1 1 ?
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_auth_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_auth_atom_id
1 covale ? ? ? ? B NAG 1 C1 L LIG . C1 L LIG 1 C1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
HETATM 1 C C1 . NAG B 1 . ? 1.0 0.0 0.0 1.0 10.0 ? 1 NAG B C1 1
"""
            + second_nag
            + f"HETATM {ligand_serial} C C1 . LIG L 2 . ? 3.0 0.0 0.0 1.0 10.0 ? 1 LIG L C1 1\n"
        )
        component = ChemicalComponent("NAG", (ChemicalAtom("C1", "C"),), frozenset())
        ligand = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset())

        structure = read_structure(path, components={"NAG": component, "LIG": ligand})
        nag_0 = structure.system.atom_index(AtomSite("B", 0, "C1"))
        nag_1 = structure.system.atom_index(AtomSite("B", 1, "C1"))
        ligand_atom = structure.system.atom_index(AtomSite("L", 0, "C1"))

        assert bool(structure.coordinate_mask[nag_0])
        assert bool(structure.coordinate_mask[nag_1]) is observe_second_nag
        assert structure.coordinates[nag_0].tolist() == [1.0, 0.0, 0.0]
        if observe_second_nag:
            assert structure.coordinates[nag_1].tolist() == [2.0, 0.0, 0.0]
        assert {
            frozenset(edge)
            for edge in structure.system.edge_index.t().tolist()
        } == {frozenset((nag_0, ligand_atom))}

    def test_unused_branch_ambiguity_is_retained_until_a_residue_uses_it(self, tmp_path: Path) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "branch-ambiguity.cif",
            """data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
1 X 2
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
B 1 X 1
B 1 X 2
""",
        )
        _, candidates = read_mmcif_branch_entities(cif_file, {"X": component})
        assert candidates[("label", "B", "1", "X", "B", None)] == frozenset({0, 1})

        row = SimpleNamespace(
            chain_id="B",
            entity_id="1",
            component_id="X",
            auth_component_id=None,
            auth_chain_id=None,
            auth_residue_id=None,
        )
        with pytest.raises(ValueError, match="maps ambiguously"):
            _resolve_branch_component_index(row, candidates, ("B", 1))

    @pytest.mark.parametrize(
        ("number", "monomer", "message"),
        [("2", "X", "is not declared"), ("1", "Y", "disagrees with declaration")],
    )
    def test_branch_scheme_num_is_authoritative_and_mon_id_must_match(
        self,
        tmp_path: Path,
        number: str,
        monomer: str,
        message: str,
    ) -> None:
        component = ChemicalComponent("X", (ChemicalAtom("A", "C"),), frozenset())
        cif_file = _read_cif(
            tmp_path / "invalid-branch-scheme.cif",
            f"""data_branch
loop_
_pdbx_entity_branch_list.entity_id
_pdbx_entity_branch_list.comp_id
_pdbx_entity_branch_list.num
1 X 1
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
B 1 {monomer} {number}
""",
        )
        with pytest.raises(ValueError, match=message):
            read_mmcif_branch_entities(cif_file, {"X": component})
