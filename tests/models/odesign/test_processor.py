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

"""Behavior of ofoldx.models.odesign.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data import Structure
from ofoldx.data.molecular import AtomSite, ComponentRef, MolecularSystem
from ofoldx.data.projection import project_arrays, project_instances
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import ContradictionError, InvalidRequestError, UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.odesign import ODesignLigandProcessor, ODesignProcessor, ODesignProcessorConfig
from ofoldx.models.odesign import processor as odesign_processor
from ofoldx.msa import MSA
from ofoldx.provenance import Basis, Resolution, state

FIRST = PositionSelection(components=(ComponentRef("A", 0),))
SECOND = PositionSelection(components=(ComponentRef("A", 1),))


def slot_names(structure, prepared) -> list[str]:
    system = structure.system
    return system.text("atom", system.atom_symbol[prepared.context.slots.atom]).tolist()


def one_hot(prepared, token: str, position: int, plane: int = 0) -> float:
    return float(prepared.features.token_features[0, position, plane * len(TOKENS) + TOKEN_INDEX[token]])


@pytest.fixture
def ligand_truth(peptide):
    target = peptide("ALA", "GLY", ligands=("ACT", "ACT"))
    context, _ = project_instances(target, np.array([True, False, False]))
    return context, target


class TestPrepare:
    @pytest.mark.parametrize("damage", ("missing", "shape", "nonfinite"))
    def test_invalid_bundled_reference_fails_explicitly(self, peptide, monkeypatch, damage):
        load = odesign_processor.load_safetensors

        def damaged(data):
            arrays = load(data)
            if damage == "missing":
                arrays.pop("GLY")
            elif damage == "shape":
                arrays["GLY"] = arrays["GLY"][:-1]
            else:
                arrays["GLY"] = arrays["GLY"].copy()
                arrays["GLY"][0, 0] = np.nan
            return arrays

        monkeypatch.setattr(odesign_processor, "load_safetensors", damaged)
        odesign_processor._representative_templates.cache_clear()
        try:
            with pytest.raises(ContradictionError, match="native_corrupt"):
                ODesignProcessor().prepare_input(peptide("ALA"))
        finally:
            odesign_processor._representative_templates.cache_clear()

    @pytest.mark.parametrize(
        ("polymer", "first", "second", "linker"),
        [
            ("polyribonucleotide", "A", "G", "N9"),
            ("polyribonucleotide", "C", "U", "N1"),
            ("polydeoxyribonucleotide", "DA", "DG", "N9"),
            ("polydeoxyribonucleotide", "DC", "DT", "N1"),
        ],
    )
    def test_designed_nucleic_hides_base_atoms(self, library, placed, polymer, first, second, linker):
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        outputs = []
        names = []
        for component in (first, second):
            system, _, _ = build_system(Recipe({"R": PolymerSequence(polymer, (component,))}, {"A": "R"}), library)
            structure = placed(system)
            prepared = processor.prepare_input(structure)
            outputs.append(prepared.features)
            names.append(slot_names(structure, prepared))
        assert names[0] == names[1]
        assert linker in names[0]
        backbone = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
        if polymer == "polyribonucleotide":
            backbone.add("O2'")
        assert set(names[0]) == backbone | {linker}
        assert not ({"C2", "C4", "C5", "C6", "C8", "N4", "N6", "N7"} & set(names[0]))
        torch.testing.assert_close(outputs[0].token_features, outputs[1].token_features, rtol=0, atol=0)
        torch.testing.assert_close(outputs[0].atom_features, outputs[1].atom_features, rtol=0, atol=0)
        torch.testing.assert_close(outputs[0].reference_positions, outputs[1].reference_positions, rtol=0, atol=0)

    def test_generated_polymer_elements_are_known_and_context_reference_is_unchanged(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = ODesignProcessor().prepare_input(structure, design=FIRST)
        slots = prepared.context.slots.atom
        tokens = prepared.context.slots.token
        elements = prepared.features.atom_features[0, :, 1:130].argmax(dim=-1).cpu().numpy()
        assert np.array_equal(elements[tokens == 0], structure.system.atomic_number[slots[tokens == 0]] - 1)
        context = tokens == 1
        reference, _ = structure.system.reference_coordinates()
        expected = reference[slots[context]]
        expected = expected - expected.mean(axis=0)
        torch.testing.assert_close(
            prepared.features.reference_positions[0, context],
            torch.tensor(expected, dtype=torch.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_nucleic_context_and_explicit_atoms_remain(self, library, placed):
        system, _, _ = build_system(
            Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "C"))}, {"A": "R", "B": "R"}), library
        )
        structure = placed(system)
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        fixed, constraint = AtomSite("A", 0, "N6"), AtomSite("A", 0, "C1'")
        prepared = processor.prepare_input(
            structure, design=FIRST, fixed_atoms=(fixed,), constraint_atoms=(constraint,)
        )
        atoms = prepared.context.slots.atom
        indices = system.atom_index([fixed, constraint, AtomSite("A", 1, "N4"), AtomSite("B", 0, "N6")])
        assert all(index in atoms for index in indices)
        fixed_slot = int(np.flatnonzero(atoms == indices[0])[0])
        assert bool(prepared.features.fixed_coordinate_mask[0, fixed_slot])
        assert not prepared.context.scope.atoms[indices[0]]
        assert not prepared.context.scope.atoms[indices[2]]
        assert not prepared.context.scope.atoms[indices[3]]
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(structure, design=FIRST, fixed_atoms=(AtomSite("A", 0, "H8"),))

    def test_explicit_constraints_require_token_centres(self, peptide):
        structure = peptide("ALA", "GLY")
        centres = (AtomSite("A", 0, "CA"), AtomSite("A", 1, "CA"))
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
        prepared = processor.prepare_input(structure, design=FIRST, constraint_atoms=centres)
        indices = structure.system.atom_index(centres)
        distance = np.linalg.norm(structure.coordinates[indices[0]] - structure.coordinates[indices[1]])
        assert float(prepared.features.constraint_distogram[0, 0, 1, 0]) == pytest.approx(distance)
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(structure, design=FIRST, constraint_atoms=(AtomSite("A", 0, "CB"),))
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(structure, design=FIRST, constraint_atoms=(AtomSite("A", 1, "N"),))
        with pytest.raises(UnsupportedError, match="model_capacity"):
            ODesignLigandProcessor().prepare_input(
                structure, generated_atoms=2, constraint_atoms=(AtomSite("A", 1, "N"),)
            )

    @pytest.mark.parametrize("unavailable", ["coordinates", "zero_occupancy"])
    def test_explicit_constraint_requires_available_coordinates(self, peptide, unavailable):
        structure = peptide("ALA", "GLY")
        atom = AtomSite("A", 0, "CA")
        index = int(structure.system.atom_index([atom])[0])
        arrays = structure.build()
        if unavailable == "coordinates":
            arrays.columns["coordinate_evidence"][index] = CoordinateEvidence.NONE
            arrays.columns["coordinates"][index] = 0
        else:
            arrays.columns["occupancy"][index] = 0
            arrays.columns["occupancy_state"][index] = state(Resolution.KNOWN, Basis.DECLARED)
        structure = Structure.from_arrays(structure.system, arrays)
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            ODesignProcessor().prepare_input(structure, design=FIRST, constraint_atoms=(atom,))
        if unavailable == "zero_occupancy":
            ODesignProcessor().prepare_input(
                structure, design=FIRST, constraint_atoms=(atom,), filter_by_occupancy=False
            )

    def test_generated_identity_hidden(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = ODesignProcessor().prepare_input(structure)
        assert [one_hot(prepared, "X", position) for position in (0, 1)] == [1.0, 1.0]
        assert [one_hot(prepared, "-", position, plane=1) for position in (0, 1)] == [1.0, 1.0]
        assert prepared.features.design_features[0].tolist() == [[1, 0, 0, 0], [1, 0, 0, 0]]
        assert set(slot_names(structure, prepared)) == {"N", "CA", "C", "O"}

    def test_context_kept_whole(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = ODesignProcessor().prepare_input(structure, design=SECOND)
        assert one_hot(prepared, "A", 0) == 1.0
        assert "CB" in slot_names(structure, prepared)
        assert prepared.context.scope.rule == "atoms"

    def test_flexible_receptor_scope(self, peptide):
        structure = peptide("ALA", "GLY")
        atoms = structure.system.component_atom_ptr
        rigid = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
        designed = slice(int(atoms[0]), int(atoms[1]))
        context = slice(int(atoms[1]), int(atoms[2]))
        assert ODesignProcessor().prepare_input(structure, design=FIRST).context.scope.atoms[context].all()
        assert not rigid.prepare_input(structure, design=FIRST).context.scope.atoms[context].any()
        assert rigid.prepare_input(structure, design=FIRST).context.scope.atoms[designed].all()

    def test_msa_profile_unpaired(self, peptide):
        pool = MSA(
            entity_id="P",
            unpaired_sequences=("AG", "AA"),
            unpaired_deletion_counts=np.zeros((2, 2), dtype=np.uint32),
            paired_sequences=("AG",),
            paired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
        )
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        prepared = ODesignProcessor().prepare_input(structure, design=FIRST, msa={"A": pool})
        assert one_hot(prepared, "A", 1, plane=1) == pytest.approx(0.5)
        assert one_hot(prepared, "G", 1, plane=1) == pytest.approx(0.5)
        assert one_hot(prepared, "-", 0, plane=1) == 1.0
        assert one_hot(prepared, "OTHER", 2, plane=1) == 1.0  # no pool covers it: its own class, not the pool's depth
        assert bool((prepared.features.msa_tokens[0, :, 0] == TOKEN_INDEX["-"]).all())

    def test_artifact_capacity(self, peptide):
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(peptide("ALA", "GLY"))

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            ODesignProcessor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = ODesignProcessor()
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure, design=FIRST)
        tensors = target
        assert tensors.coordinates.shape == (1, len(prepared.context.slots.atom), 3)
        assert tensors.coordinate_mask.all()
        assert bool(tensors.covalent_pair_mask.all())

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_distinct_truth(self, peptide, dtype):
        structure = peptide("ALA", "GLY")
        source = structure.build()
        source.columns["coordinates"] += [0.123456789123, -0.987654321987, 0.314159265359]
        structure = Structure.from_arrays(structure.system, source)
        arrays = structure.build()
        arrays.columns["coordinates"] += [15, -4, 8]
        target = Structure.from_arrays(structure.system, arrays)
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"), dtype=dtype)
        prepared, truth = processor.prepare(structure, target=target, design=FIRST)
        tensors = truth
        fixed = prepared.features.fixed_coordinate_mask[0]
        torch.testing.assert_close(
            tensors.coordinates[0, fixed],
            torch.tensor(structure.coordinates[prepared.context.slots.atom[fixed.numpy()]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            tensors.coordinates[0, ~fixed],
            torch.tensor(target.coordinates[prepared.context.slots.atom[~fixed]], dtype=torch.float64),
        )

    @pytest.mark.parametrize("filter_by_occupancy", [True, False])
    def test_separate_target_reuses_observation_policy(self, peptide, filter_by_occupancy):
        structure = peptide("ALA", "GLY")
        arrays = structure.build()
        arrays.columns["occupancy"][0] = 0
        arrays.columns["occupancy_state"][0] = state(Resolution.KNOWN, Basis.DECLARED)
        target = Structure.from_arrays(structure.system, arrays)
        processor = ODesignProcessor()
        combined, labels = processor.prepare(
            structure, target=target, design=FIRST, filter_by_occupancy=filter_by_occupancy
        )
        prepared = processor.prepare_input(structure, design=FIRST, filter_by_occupancy=filter_by_occupancy)
        separate = processor.prepare_target(target, prepared=prepared)
        torch.testing.assert_close(separate.to_dict(), labels.to_dict())
        slot = int(np.flatnonzero(combined.context.slots.atom == 0)[0])
        assert bool(separate.coordinate_mask[0, slot]) is not filter_by_occupancy


class TestLigandPrepare:
    def test_generated_slots_unbound(self, peptide):
        prepared = ODesignLigandProcessor().prepare_input(peptide("ALA", "GLY"), generated_atoms=3)
        assert prepared.context.slots.atom[-3:].tolist() == [-1, -1, -1]
        assert int(prepared.features.generated_atom_mask.sum()) == 3
        assert prepared.features.atom_features[0, -3:, 1:130].argmax(dim=-1).tolist() == [128, 128, 128]
        assert prepared.context.scope.rule == "none"
        assert one_hot(prepared, "OTHER", -1) == 1.0

    def test_empty_context_generates(self, polymers):
        prepared = ODesignLigandProcessor().prepare_input(Structure.unobserved(polymers()), generated_atoms=3)
        assert prepared.context.slots.atom.tolist() == [-1, -1, -1]

    def test_sampled_count_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            ODesignLigandProcessor().prepare_input(peptide("ALA"), generated_atoms=range(2, 4))

    @pytest.mark.parametrize("filter_by_occupancy", [True, False])
    def test_training_truth(self, ligand_truth, filter_by_occupancy):
        context, original = ligand_truth
        system, arrays, mapping = project_arrays(
            original.system.build(), original.build(), np.array([2, 0, 1]), np.ones(len(original.coordinates), bool)
        )
        assert arrays is not None
        generated = mapping.atom[context.system.component_atom_ptr[-1] :]
        rows = np.concatenate((generated[len(generated) // 2 :], generated[: len(generated) // 2]))
        arrays.columns["coordinates"] += [20, -7, 5]
        arrays.columns["occupancy"][rows[0]] = 0
        arrays.columns["occupancy_state"][rows[0]] = state(Resolution.KNOWN, Basis.DECLARED)
        arrays.columns["coordinate_evidence"][rows[-1]] = CoordinateEvidence.NONE
        arrays.columns["coordinates"][rows[-1]] = 0
        target = Structure.from_arrays(MolecularSystem.from_arrays(system), arrays)
        processor = ODesignLigandProcessor()
        prepared, truth = processor.prepare(
            context,
            target=target,
            generated_target_instance_ids=("C", "B"),
            generated_atoms=range(len(rows), len(rows) + 1),
            filter_by_occupancy=filter_by_occupancy,
        )
        separate_input = processor.prepare_input(
            context, generated_atoms=len(rows), filter_by_occupancy=filter_by_occupancy
        )
        separate_truth = processor.prepare_target(
            target, prepared=separate_input, generated_target_instance_ids=("C", "B")
        )
        torch.testing.assert_close(separate_truth.to_dict(), truth.to_dict())
        tensors = truth
        count = len(prepared.context.slots.atom) - len(rows)
        assert prepared.context.structure is context
        assert (prepared.context.slots.atom[-len(rows) :] == -1).all()
        np.testing.assert_allclose(tensors.coordinates[0, :count], prepared.features.observed_positions[0, :count])
        assert bool(tensors.coordinate_mask[0, count]) is not filter_by_occupancy
        assert not tensors.coordinate_mask[0, -1]
        np.testing.assert_allclose(tensors.coordinates[0, count + 1 :], target.coordinates[rows[1:]])
        slot_of = {int(atom): count + slot for slot, atom in enumerate(rows)}
        expected = {
            (slot_of[int(first)], slot_of[int(second)])
            for first, second in target.system.edge_index.T
            if int(first) in slot_of and int(second) in slot_of
        }
        actual = {tuple(edge) for edge in tensors.covalent_bond_index[0].tolist() if min(edge) >= count}
        assert expected == actual and expected
        assert tensors.covalent_pair_mask[0, count:, count:].all()

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_fixed_condition_preserves_source_precision(self, ligand_truth, dtype):
        context, target = ligand_truth
        source = context.build()
        source.columns["coordinates"] += [0.123456789123, -0.987654321987, 0.314159265359]
        context = Structure.from_arrays(context.system, source)
        changed = target.build()
        changed.columns["coordinates"] += [13, 17, 23]
        target = Structure.from_arrays(target.system, changed)
        processor = ODesignLigandProcessor(dtype=dtype)
        prepared, truth = processor.prepare(context, target=target, generated_target_instance_ids=("B",))
        fixed = prepared.features.fixed_coordinate_mask[0]
        assert fixed.any()
        torch.testing.assert_close(
            truth.coordinates[0, fixed],
            torch.tensor(context.coordinates[prepared.context.slots.atom[fixed.numpy()]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )

    def test_truth_stays_separate(self, ligand_truth):
        context, target = ligand_truth
        processor = ODesignLigandProcessor()
        prepared, truth = processor.prepare(context, target=target, generated_target_instance_ids=("B",))
        arrays = target.build()
        arrays.columns["coordinates"] += [50, -11, 3]
        changed = Structure.from_arrays(target.system, arrays)
        other, other_truth = processor.prepare(context, target=changed, generated_target_instance_ids=("B",))
        for name, value in prepared.features.items():
            torch.testing.assert_close(value, getattr(other.features, name))
        generated = prepared.features.generated_atom_mask[0].numpy()
        np.testing.assert_allclose(
            other_truth.coordinates[0, generated] - truth.coordinates[0, generated],
            np.tile([50, -11, 3], (int(generated.sum()), 1)),
        )

    @pytest.mark.parametrize("generated_atoms", [1, range(1, 3)])
    def test_training_count(self, ligand_truth, generated_atoms):
        context, target = ligand_truth
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            ODesignLigandProcessor().prepare(
                context, target=target, generated_target_instance_ids=("B",), generated_atoms=generated_atoms
            )


class TestLigandPreparationInterface:
    def test_composition_reuses_input_task_size(self, ligand_truth):
        context, truth = ligand_truth
        processor = ODesignLigandProcessor()
        combined, labels = processor.prepare(context, target=truth, generated_target_instance_ids=("B",))
        count = int(combined.features.generated_atom_mask.sum())
        prepared = processor.prepare_input(context, generated_atoms=count)
        target = processor.prepare_target(truth, prepared=prepared, generated_target_instance_ids=("B",))
        torch.testing.assert_close(prepared.features.to_dict(), combined.features.to_dict())
        torch.testing.assert_close(target.to_dict(), labels.to_dict())
        input_only, no_target = processor.prepare(context, generated_atoms=count)
        torch.testing.assert_close(input_only.features.to_dict(), prepared.features.to_dict())
        assert no_target is None
        wrong = processor.prepare_input(context, generated_atoms=count + 1)
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            processor.prepare_target(truth, prepared=wrong, generated_target_instance_ids=("B",))


@pytest.mark.parametrize(
    ("polymer", "source_sequence", "target_sequence"),
    [
        ("polyribonucleotide", ("N", "N"), ("A", "C")),
        ("polyribonucleotide", ("A", "C"), ("C", "A")),
        ("polydeoxyribonucleotide", ("DN", "DN"), ("DA", "DC")),
        ("polydeoxyribonucleotide", ("DA", "DC"), ("DC", "DA")),
    ],
)
@pytest.mark.parametrize("truth_state", ["observed", "missing", "zero_occupancy", "include_zero"])
def test_nucleic_target_linker_matches_observed_truth(
    library, placed, polymer, source_sequence, target_sequence, truth_state
):
    source_system, _, _ = build_system(Recipe({"R": PolymerSequence(polymer, source_sequence)}, {"A": "R"}), library)
    target_system, _, _ = build_system(Recipe({"R": PolymerSequence(polymer, target_sequence)}, {"A": "R"}), library)
    source = Structure.unobserved(source_system)
    target = placed(target_system)
    first_name = "N9" if target_sequence[0] in ("A", "G", "DA", "DG") else "N1"
    first_atom = int(target_system.atom_index([AtomSite("A", 0, first_name)])[0])
    arrays = target.build()
    if truth_state == "missing":
        arrays.columns["coordinates"][first_atom] = 0
        arrays.columns["coordinate_evidence"][first_atom] = CoordinateEvidence.NONE
    if truth_state in ("zero_occupancy", "include_zero"):
        arrays.columns["occupancy"][first_atom] = 0
        arrays.columns["occupancy_state"][first_atom] = state(Resolution.KNOWN, Basis.DECLARED)
    target = Structure.from_arrays(target_system, arrays)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared, truth = processor.prepare(source, target=target, filter_by_occupancy=truth_state != "include_zero")
    slots = prepared.context.slots
    for component, identity in enumerate(target_sequence):
        token_rows = np.flatnonzero(slots.token == component)
        if source_sequence[component] in ("N", "DN"):
            linker_row = int(token_rows[slots.atom[token_rows] < 0][0])
        else:
            source_name = "N9" if source_sequence[component] in ("A", "G", "DA", "DG") else "N1"
            source_atom = int(source_system.atom_index([AtomSite("A", component, source_name)])[0])
            linker_row = int(np.flatnonzero(slots.atom == source_atom)[0])
        truth_name = "N9" if identity in ("A", "G", "DA", "DG") else "N1"
        truth_atom = int(target_system.atom_index([AtomSite("A", component, truth_name)])[0])
        expected = not (component == 0 and truth_state in ("missing", "zero_occupancy"))
        assert bool(truth.coordinate_mask[0, linker_row]) == expected
        if expected:
            np.testing.assert_array_equal(
                truth.coordinates[0, linker_row].cpu().numpy(), target.coordinates[truth_atom]
            )
        if source_sequence[component] in ("N", "DN"):
            anchor = int(source_system.atom_index([AtomSite("A", component, "C1'")])[0])
            anchor_row = int(np.flatnonzero(slots.atom == anchor)[0])
            bonds = truth.covalent_bond_index[0].cpu().numpy()
            assert any(set(pair) == {anchor_row, linker_row} for pair in bonds)
            assert bool(truth.covalent_pair_mask[0, anchor_row, linker_row])


@pytest.mark.parametrize("case", ["missing_component", "unknown_identity", "unknown_bond"])
def test_nucleic_target_linker_requires_truth_identity_and_topology(library, placed, case):
    source_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("N", "N"))}, {"A": "R"}), library
    )
    target_sequence = (
        ("A",) if case == "missing_component" else (("N", "A") if case == "unknown_identity" else ("A", "A"))
    )
    target_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", target_sequence)}, {"A": "R"}), library
    )
    target = placed(target_system)
    component = 1 if case == "missing_component" else 0
    if case == "unknown_bond":
        atom, anchor = target_system.atom_index([AtomSite("A", component, "N9"), AtomSite("A", component, "C1'")])
        arrays = target_system.build()
        edge = arrays.columns["edge_index"]
        bond = np.flatnonzero(((edge[0] == atom) & (edge[1] == anchor)) | ((edge[1] == atom) & (edge[0] == anchor)))
        assert len(bond) == 1
        arrays.columns["bond_kind"][bond] = 0
        arrays.columns["bond_kind_state"][bond] = state(Resolution.UNKNOWN, Basis.NONE)
        arrays.columns["component_topology_complete"][component] = False
        target_system = MolecularSystem.from_arrays(arrays)
        target = Structure.from_arrays(target_system, target.build())
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared, truth = processor.prepare(Structure.unobserved(source_system), target=target)
    slots = prepared.context.slots
    row = int(np.flatnonzero((slots.token == component) & (slots.atom < 0))[0])
    assert not bool(truth.coordinate_mask[0, row])
    assert not bool(truth.covalent_pair_mask[0, row].any())
    assert not bool((truth.covalent_bond_index[0] == row).any())


def test_nucleic_target_linker_keeps_fixed_source_coordinate(library, placed):
    source_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "A"))}, {"A": "R"}), library
    )
    target_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("C", "C"))}, {"A": "R"}), library
    )
    source, target = placed(source_system), placed(target_system)
    fixed = AtomSite("A", 0, "N9")
    atom = int(source_system.atom_index([fixed])[0])
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared, truth = processor.prepare(source, target=target, fixed_atoms=(fixed,))
    row = int(np.flatnonzero(prepared.context.slots.atom == atom)[0])
    assert bool(truth.coordinate_mask[0, row])
    np.testing.assert_array_equal(truth.coordinates[0, row].cpu().numpy(), source.coordinates[atom])


@pytest.mark.parametrize(
    "uncertainty",
    ["linker_fate", "anchor_fate", "linker_endpoint", "anchor_endpoint", "linker_conflict", "anchor_conflict"],
)
def test_nucleic_target_linker_needs_determined_truth_atoms(library, placed, uncertainty):
    from ofoldx.data.molecular import AtomFate, EndpointState

    source_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("N", "N"))}, {"A": "R"}), library
    )
    target_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "C"))}, {"A": "R"}), library
    )
    target = placed(target_system)
    linker, anchor = target_system.atom_index([AtomSite("A", 0, "N9"), AtomSite("A", 0, "C1'")])
    arrays = target_system.build()
    atom = linker if uncertainty.startswith("linker") else anchor
    if uncertainty.endswith("fate"):
        arrays.columns["atom_fate"][atom] = AtomFate.UNDETERMINED
        arrays.columns["component_heavy_atoms_complete"][0] = False
    else:
        arrays.columns["atom_endpoint_state"][atom] = (
            EndpointState.CONFLICT if uncertainty.endswith("conflict") else EndpointState.UNKNOWN
        )
    arrays.columns["component_topology_complete"][0] = False
    target_system = MolecularSystem.from_arrays(arrays)
    target = Structure.from_arrays(target_system, target.build())
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared, truth = processor.prepare(Structure.unobserved(source_system), target=target)
    row = int(np.flatnonzero((prepared.context.slots.token == 0) & (prepared.context.slots.atom < 0))[0])
    assert not bool(truth.coordinate_mask[0, row])
    assert not bool(truth.covalent_pair_mask[0, row].any())
    assert not bool((truth.covalent_bond_index[0] == row).any())


@pytest.mark.parametrize(
    ("polymer", "modified", "standard"),
    [("polyribonucleotide", "I", "A"), ("polydeoxyribonucleotide", "DI", "DA")],
)
def test_nonstandard_nucleic_can_be_context_but_not_designed(library, inosine_components, polymer, modified, standard):
    definitions = inosine_components
    recipe = Recipe(
        {"M": PolymerSequence(polymer, (modified,)), "S": PolymerSequence(polymer, (standard,))},
        {"A": "M", "B": "S"},
    )
    system, _, _ = build_system(recipe, library, definitions)
    source = Structure.unobserved(system)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    with pytest.raises(UnsupportedError, match="model_capacity"):
        processor.prepare_input(source, design=PositionSelection(instances=("A",)))
    processor.prepare_input(source, design=PositionSelection(instances=("B",)))
