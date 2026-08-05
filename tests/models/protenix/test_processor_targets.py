from __future__ import annotations

import torch

from ofoldx.data import (
    AtomSite,
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    MoleculeBond,
    PolymerSequence,
    Structure,
    SystemBond,
)
from ofoldx.models.protenix.config import ProtenixProcessorConfig
from ofoldx.models.protenix.processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
    _ligand_polymer_bond_features,
    _local_symmetry_features,
    _molecule_metadata,
    _permutation_pool_mask,
)


def _ligand(component_id: str = "LIG", *, complete: bool = True) -> Molecule:
    atoms = (
        ChemicalAtom("C0", "C"),
        ChemicalAtom("O1", "O"),
    ) + ((ChemicalAtom("O2", "O"),) if complete else ())
    bonds = {ChemicalBond("C0", "O1")}
    if complete:
        bonds.add(ChemicalBond("C0", "O2"))
    coordinates = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) + (
        ((0.0, 1.0, 0.0),) if complete else ()
    )
    return Molecule(
        (
            ChemicalComponent(
                component_id,
                atoms,
                frozenset(bonds),
                reference_coordinates=coordinates,
            ),
        )
    )


def _mixed_structure(
    *,
    coordinate_mask: torch.Tensor | None = None,
    resolution: torch.Tensor | None = None,
    resolution_is_known: torch.Tensor | None = None,
) -> Structure:
    protein = PolymerSequence.from_one_letter("D")
    ligand = _ligand()
    system = MolecularSystem.from_entities(
        {"protein": protein, "ligand": ligand},
        {"A": "protein", "B": "protein", "L1": "ligand", "L2": "ligand"},
        (
            SystemBond(AtomSite("A", 0, "CB"), AtomSite("L1", 0, "C0")),
            SystemBond(AtomSite("B", 0, "CB"), AtomSite("L2", 0, "C0")),
        ),
    )
    coordinates = system.reference_coordinates.detach().cpu().clone()
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel()),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel()),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    coordinates[:, 0] += component_instance[atom_component].float() * 10.0
    component_count = system.component_symbol.numel()
    atom_count = system.atom_symbol.numel()
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=(
            torch.ones(atom_count, dtype=torch.bool) if coordinate_mask is None else coordinate_mask
        ),
        occupancy=torch.ones(atom_count),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=tuple(str(index) for index in range(component_count)),
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1, dtype=torch.long),
        resolution=resolution,
        resolution_is_known=resolution_is_known,
        batch_size=[],
    )


def _config() -> ProtenixProcessorConfig:
    return ProtenixProcessorConfig(augment_reference_positions=False)


def _has_local_swap(
    system: MolecularSystem,
    first: AtomSite,
    second: AtomSite,
    *,
    atom_indices: tuple[int, ...] | None = None,
) -> bool:
    if atom_indices is None:
        atom_indices = tuple(
            index
            for index, atomic_number in enumerate(system.atomic_number.tolist())
            if atomic_number != 1
        )
    system_to_model = {canonical: current for current, canonical in enumerate(atom_indices)}
    first_index = system_to_model[system.atom_index(first)]
    second_index = system_to_model[system.atom_index(second)]
    symmetry_atom_index, symmetry_permutation = _local_symmetry_features(system, atom_indices)
    for group_index, atom_group in enumerate(symmetry_atom_index):
        first_position = torch.nonzero(atom_group == first_index, as_tuple=False).flatten()
        second_position = torch.nonzero(atom_group == second_index, as_tuple=False).flatten()
        if first_position.numel() != 1 or second_position.numel() != 1:
            continue
        for permutation in symmetry_permutation[group_index]:
            if (
                permutation[int(first_position[0])] == second_index
                and permutation[int(second_position[0])] == first_index
            ):
                return True
    return False


def test_molecule_metadata_uses_covalent_components_and_exact_layout() -> None:
    structure = _mixed_structure()
    system = structure.system
    molecule_id, equivalence, molecule_atom_index, _ = _molecule_metadata(
        system,
        structure.coordinate_mask,
    )

    first_atoms = [
        system.atom_index(AtomSite("A", 0, "CA")),
        system.atom_index(AtomSite("L1", 0, "O1")),
    ]
    second_atoms = [
        system.atom_index(AtomSite("B", 0, "CA")),
        system.atom_index(AtomSite("L2", 0, "O1")),
    ]
    assert molecule_id[first_atoms].unique().numel() == 1
    assert molecule_id[second_atoms].unique().numel() == 1
    assert molecule_id[first_atoms[0]] != molecule_id[second_atoms[0]]
    assert equivalence[first_atoms[0]] == equivalence[second_atoms[0]]
    assert molecule_atom_index[first_atoms[0]] == molecule_atom_index[second_atoms[0]]

    declared = _ligand("VAR", complete=True)
    incomplete = _ligand("VAR", complete=False)
    varied = MolecularSystem.from_entities(
        {"ligand": declared},
        {"L1": "ligand", "L2": "ligand"},
        instance_values={"L2": incomplete},
    )
    _, varied_equivalence, _, _ = _molecule_metadata(
        varied,
        torch.ones(varied.atom_symbol.numel(), dtype=torch.bool),
    )
    first = varied.atom_index(AtomSite("L1", 0, "C0"))
    second = varied.atom_index(AtomSite("L2", 0, "C0"))
    assert varied_equivalence[first] != varied_equivalence[second]


def test_molecule_metadata_uses_resolved_instance_graph_and_ignores_disulfides() -> None:
    cysteine = PolymerSequence.from_one_letter("C")
    disulfide = MolecularSystem.from_entities(
        {"protein": cysteine},
        {"A": "protein", "B": "protein"},
        (SystemBond(AtomSite("A", 0, "SG"), AtomSite("B", 0, "SG")),),
    )
    disulfide_mask = torch.ones(disulfide.atom_symbol.numel(), dtype=torch.bool)
    molecule_id, _, _, _ = _molecule_metadata(disulfide, disulfide_mask)
    assert molecule_id[disulfide.atom_index(AtomSite("A", 0, "CA"))] != molecule_id[
        disulfide.atom_index(AtomSite("B", 0, "CA"))
    ]

    first_component = ChemicalComponent(
        "ONE",
        (ChemicalAtom("C", "C"),),
        frozenset(),
        reference_coordinates=((0.0, 0.0, 0.0),),
    )
    second_component = ChemicalComponent(
        "TWO",
        (ChemicalAtom("O", "O"),),
        frozenset(),
        reference_coordinates=((1.0, 0.0, 0.0),),
    )
    disconnected = MolecularSystem.from_entities(
        {"ligand": Molecule((first_component, second_component))},
        {"L": "ligand"},
    )
    disconnected_mask = torch.ones(disconnected.atom_symbol.numel(), dtype=torch.bool)
    disconnected_id, _, disconnected_atom_index, _ = _molecule_metadata(
        disconnected,
        disconnected_mask,
    )
    assert disconnected_id.unique().tolist() == [0]
    assert disconnected_atom_index.tolist() == [0, 1]

    mixed = _mixed_structure()
    unresolved = mixed.coordinate_mask.clone()
    unresolved[mixed.system.atom_index(AtomSite("L2", 0, "C0"))] = False
    mixed_id, _, _, _ = _molecule_metadata(mixed.system, unresolved)
    assert mixed_id[mixed.system.atom_index(AtomSite("A", 0, "CA"))] == mixed_id[
        mixed.system.atom_index(AtomSite("L1", 0, "O1"))
    ]
    assert mixed_id[mixed.system.atom_index(AtomSite("B", 0, "CA"))] != mixed_id[
        mixed.system.atom_index(AtomSite("L2", 0, "O1"))
    ]


def test_permutation_pool_expands_continuous_but_restricts_spatial_candidates() -> None:
    structure = _mixed_structure()
    system = structure.system
    molecule_id, equivalence, molecule_atom_index, _ = _molecule_metadata(
        system,
        structure.coordinate_mask,
    )
    first = system.atom_index(AtomSite("A", 0, "CA"))
    second = system.atom_index(AtomSite("B", 0, "CA"))

    continuous = _permutation_pool_mask(
        (first,),
        molecule_id,
        equivalence,
        molecule_atom_index,
        restrict_to_current_molecules=False,
    )
    spatial = _permutation_pool_mask(
        (first,),
        molecule_id,
        equivalence,
        molecule_atom_index,
        restrict_to_current_molecules=True,
    )
    assert continuous[first] and continuous[second]
    assert spatial[first] and not spatial[second]


def test_local_symmetry_uses_rdkit_for_standard_arg_and_nonstandard_carboxylate() -> None:
    for residue, first, second in (
        ("V", "CG1", "CG2"),
        ("L", "CD1", "CD2"),
        ("R", "NH1", "NH2"),
    ):
        standard = MolecularSystem.from_entities(
            {"protein": PolymerSequence.from_one_letter(residue)},
            {"A": "protein"},
        )
        assert _has_local_swap(
            standard,
            AtomSite("A", 0, first),
            AtomSite("A", 0, second),
        )

    carboxylate = Molecule.from_smiles(
        "CC(=O)[O-]",
        atom_ids=("C0", "C1", "O1", "O2"),
    )
    carboxylate_system = MolecularSystem.from_entities(
        {"ligand": carboxylate},
        {"L": "ligand"},
    )
    assert _has_local_swap(
        carboxylate_system,
        AtomSite("L", 0, "O1"),
        AtomSite("L", 0, "O2"),
    )


def test_local_symmetry_groups_remain_per_component_for_resolved_thresholds() -> None:
    carboxylate = Molecule.from_smiles(
        "CC(=O)[O-]",
        atom_ids=("C0", "C1", "O1", "O2"),
    )
    system = MolecularSystem.from_entities(
        {"ligand": carboxylate},
        {"L1": "ligand", "L2": "ligand"},
    )
    atom_indices = tuple(range(system.atom_symbol.numel()))
    symmetry_atom_index, _ = _local_symmetry_features(system, atom_indices)
    expected_groups = {
        frozenset(
            system.atom_index(AtomSite(instance, 0, atom_name))
            for atom_name in ("C0", "C1", "O1", "O2")
        )
        for instance in ("L1", "L2")
    }
    actual_groups = {
        frozenset(group[group >= 0].tolist()) for group in symmetry_atom_index
    }
    assert actual_groups == expected_groups


def test_local_symmetry_preserves_chirality_and_component_boundaries() -> None:
    stereoisomer = Molecule.from_smiles(
        "F[C@H](Cl)[C@@H](F)Cl",
        atom_ids=("F1", "C1", "CL1", "C2", "F2", "CL2"),
    )
    stereoisomer_system = MolecularSystem.from_entities(
        {"ligand": stereoisomer},
        {"S": "ligand"},
    )
    assert not _has_local_swap(
        stereoisomer_system,
        AtomSite("S", 0, "C1"),
        AtomSite("S", 0, "C2"),
    )

    chiral = Molecule.from_smiles(
        "N[C@@H](C)C(=O)[O-]",
        atom_ids=("N", "CA", "CB", "C", "O1", "O2"),
    )
    chiral_system = MolecularSystem.from_entities(
        {"ligand": chiral},
        {"L": "ligand"},
    )
    assert _has_local_swap(
        chiral_system,
        AtomSite("L", 0, "O1"),
        AtomSite("L", 0, "O2"),
    )
    atom_indices = tuple(range(chiral_system.atom_symbol.numel()))
    symmetry_atom_index, symmetry_permutation = _local_symmetry_features(
        chiral_system,
        atom_indices,
    )
    chiral_center = chiral_system.atom_index(AtomSite("L", 0, "CA"))
    for group_index, atom_group in enumerate(symmetry_atom_index):
        center_position = torch.nonzero(atom_group == chiral_center, as_tuple=False).flatten()
        if center_position.numel() == 1:
            assert torch.all(symmetry_permutation[group_index, :, int(center_position[0])] == chiral_center)

    anchor = ChemicalComponent(
        "ANC",
        (ChemicalAtom("X", "C"),),
        frozenset(),
        reference_coordinates=((0.0, 0.0, 0.0),),
    )
    connected = Molecule(
        (chiral.components[0], anchor),
        frozenset({MoleculeBond(0, "O1", 1, "X")}),
    )
    connected_system = MolecularSystem.from_entities(
        {"ligand": connected},
        {"L": "ligand"},
    )
    assert not _has_local_swap(
        connected_system,
        AtomSite("L", 0, "O1"),
        AtomSite("L", 0, "O2"),
    )
    component_stop = int(connected_system.component_atom_ptr[1])
    assert _has_local_swap(
        connected_system,
        AtomSite("L", 0, "O1"),
        AtomSite("L", 0, "O2"),
        atom_indices=tuple(range(component_stop)),
    )


def test_ligand_polymer_bond_ion_detection_uses_current_crop_axis() -> None:
    system = _mixed_structure().system
    polymer = system.atom_index(AtomSite("A", 0, "CB"))
    ligand_bond = system.atom_index(AtomSite("L1", 0, "C0"))
    ligand_other = system.atom_index(AtomSite("L1", 0, "O1"))

    ion_index, ion_mask = _ligand_polymer_bond_features(system, (polymer, ligand_bond))
    assert ion_index.shape == (0, 2)
    assert ion_mask.shape == (0,)

    bond_index, bond_mask = _ligand_polymer_bond_features(
        system,
        (polymer, ligand_bond, ligand_other),
    )
    assert bond_index.tolist() == [[0, 1]]
    assert bond_mask.tolist() == [True]


def test_training_target_populates_frames_representatives_bonds_and_symmetry() -> None:
    structure = _mixed_structure()
    model_input, target = ProtenixProcessor(_config()).process_training(structure)
    system_to_model = {
        canonical: current for current, canonical in enumerate(model_input.atom_index[0].tolist())
    }

    assert target.molecule_id is not None
    assert target.molecule_equivalence_index is not None
    assert target.molecule_atom_index is not None
    assert target.representative_atom_mask is not None
    assert target.permutation_pool_mask is not None
    assert target.atom_molecule_type is not None
    assert target.symmetry_atom_index is not None
    assert target.symmetry_permutation is not None
    assert target.token_center_atom_index is not None
    assert target.token_frame_atom_index is not None
    assert target.token_has_frame is not None
    assert target.plddt_rep_atom_mask is not None
    assert target.ligand_polymer_bond_index is not None
    assert target.ligand_polymer_bond_mask is not None
    assert target.resolution is not None
    assert target.resolution_is_known is not None

    atom_count = structure.system.atom_symbol.numel()
    assert target.molecule_id.shape == (1, atom_count)
    expected_pool = torch.zeros((1, atom_count), dtype=torch.bool)
    expected_pool[0, model_input.atom_index[0]] = True
    assert torch.equal(target.permutation_pool_mask, expected_pool)
    assert target.token_center_atom_index.shape == model_input.token_index.shape
    assert target.token_frame_atom_index.shape == (*model_input.token_index.shape, 3)
    assert target.token_has_frame.all()
    assert target.resolution.tolist() == [-1.0]
    assert target.resolution_is_known.tolist() == [False]

    first_frame = [
        system_to_model[structure.system.atom_index(AtomSite("A", 0, name))]
        for name in ("N", "CA", "C")
    ]
    assert target.token_frame_atom_index[0, 0].tolist() == first_frame
    assert target.token_center_atom_index[0, 0] == first_frame[1]

    expected_plddt = {
        system_to_model[structure.system.atom_index(AtomSite(instance, 0, "CA"))]
        for instance in ("A", "B")
    }
    assert set(torch.nonzero(target.plddt_rep_atom_mask[0], as_tuple=False).flatten().tolist()) == expected_plddt
    assert target.ligand_polymer_bond_index.shape == (1, 2, 2)
    assert target.ligand_polymer_bond_mask.tolist() == [[True, True]]

    for group_index, atom_group in enumerate(target.symmetry_atom_index[0]):
        identity = target.symmetry_permutation[0, group_index, 0]
        valid = atom_group >= 0
        assert torch.equal(atom_group[valid], identity[valid])
        assert torch.equal(identity[~valid], torch.full_like(identity[~valid], -1))
    assert bool((target.symmetry_atom_index == -1).any())


def test_all_processor_variants_publish_the_same_baseline_target_contract() -> None:
    structure = _mixed_structure()
    processors = (
        ProtenixProcessor(_config()),
        ProtenixTemplateProcessor(_config()),
        ProtenixStructureConditionProcessor(_config()),
        ProtenixConstraintProcessor(_config()),
    )
    targets = [processor.process_training(structure)[1] for processor in processors]

    for target in targets:
        assert target.molecule_id is not None
        assert target.symmetry_atom_index is not None
        assert target.token_frame_atom_index is not None
        assert target.ligand_polymer_bond_index is not None
        assert target.resolution_is_known is not None
        assert torch.equal(targets[0].molecule_id, target.molecule_id)
        assert torch.equal(targets[0].symmetry_atom_index, target.symmetry_atom_index)
        assert torch.equal(targets[0].token_frame_atom_index, target.token_frame_atom_index)
        assert torch.equal(targets[0].ligand_polymer_bond_index, target.ligand_polymer_bond_index)


def test_training_target_preserves_known_resolution_metadata() -> None:
    structure = _mixed_structure(
        resolution=torch.tensor(2.4),
        resolution_is_known=torch.tensor(True),
    )
    _, target = ProtenixProcessor(_config()).process_training(structure)

    assert target.resolution is not None
    assert target.resolution_is_known is not None
    torch.testing.assert_close(target.resolution, torch.tensor([2.4]))
    assert target.resolution_is_known.tolist() == [True]


def test_training_target_remains_float32_for_bfloat16_processor() -> None:
    structure = _mixed_structure(
        resolution=torch.tensor(2.4),
        resolution_is_known=torch.tensor(True),
    )
    model_input, target = ProtenixProcessor(
        _config(),
        dtype=torch.bfloat16,
    ).process_training(structure)

    assert model_input.reference_positions.dtype == torch.bfloat16
    assert target.coordinates.dtype == torch.float32
    assert target.resolution is not None
    assert target.resolution.dtype == torch.float32
    torch.testing.assert_close(target.coordinates[0], structure.coordinates.float())
