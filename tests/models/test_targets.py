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

"""Behavior of ofoldx.models.targets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import MolecularSystem, MonomerType
from ofoldx.data.projection import project_arrays, project_atoms
from ofoldx.data.structure import CoordinateEvidence, ResolutionKind, Structure, replace_coordinates
from ofoldx.data.tokenization import select_tokens
from ofoldx.models.alphafold3 import AlphaFold3Processor
from ofoldx.models.boltz1 import Boltz1Processor
from ofoldx.models.boltz2 import Boltz2Processor
from ofoldx.models.boltzgen import BoltzGenProcessor
from ofoldx.models.layout import LayoutTable, slot_map, token_layout
from ofoldx.models.odesign import ODesignProcessor
from ofoldx.models.preparation import PreparationContext, PreparationRecord, PreparedInput, RegenerationScope
from ofoldx.models.protenix import ProtenixProcessor
from ofoldx.models.targets import SequenceTarget, SymmetryPolicy, TruthState, align_target_components, prepare_target
from ofoldx.provenance import Basis, Records, Resolution, state

PROTEIN = MonomerType.PROTEIN
TABLE = LayoutTable(
    family="test",
    monomer_types={PROTEIN: 0},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="nonstandard",
    exclude_designated_leaving=True,
    centre_atom={PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)
POLICY = SymmetryPolicy(components="ligands", resonance=False, pin_unobserved=True)


@pytest.fixture
def task():
    def prepare(structure, *, cropped=False):
        layout = token_layout(structure.system, TABLE)
        if cropped:
            layout = select_tokens(layout, np.arange(len(layout.token_component)) == 0)
        return PreparedInput(
            None,
            PreparationContext(
                structure,
                slot_map(structure.system, layout, TABLE),
                RegenerationScope.all(structure.system),
                PreparationRecord(Records.empty(), np.zeros(len(structure.system.component_symbol), dtype=np.uint8)),
            ),
        )

    return prepare


class TestPrepareTarget:
    def test_truth_states(self, peptide, task):
        structure = peptide("ALA", ligands=("ACT",))
        hidden = np.zeros(len(structure.coordinates), dtype=bool)
        hidden[1] = True
        unobserved, _ = replace_coordinates(
            structure, np.zeros(0, dtype=np.int64), np.zeros((0, 3)), scope=hidden, evidence=CoordinateEvidence.OBSERVED
        )
        prepared = task(unobserved)
        target = prepare_target(prepared, unobserved, symmetry=POLICY)
        assert target.truth[1] == TruthState.UNOBSERVED
        assert (target.truth[prepared.context.slots.unaddressed] == TruthState.EXCLUDED_BY_LAYOUT).all()
        assert (target.coordinates[target.truth != TruthState.OBSERVED] == 0).all()

    @pytest.mark.parametrize("case", ["reordered", "missing", "cropped", "unoccupied"])
    def test_independent_truth(self, polymers, task, case):
        source = Structure.unobserved(polymers(("A", "P", ("ALA",)), ("B", "P", ("ALA",))))
        arrays = polymers(("B", "P", ("ALA",)), ("A", "P", ("ALA",))).build()
        arrays.columns["atom_symbol"][:] = [1, 0, 1, 0]
        truth = Structure.unobserved(MolecularSystem.from_arrays(arrays))
        truth, _ = replace_coordinates(
            truth,
            np.arange(4),
            np.array([[40, 0, 0], [30, 0, 0], [20, 0, 0], [10, 0, 0]]),
            scope=np.ones(4, dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        expected = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]], dtype=float)
        states = np.full(4, TruthState.OBSERVED)
        if case == "missing":
            truth, _ = project_atoms(truth, np.array([True, True, False, True]))
            states[1] = TruthState.UNOBSERVED
        elif case == "cropped":
            states[2:] = TruthState.EXCLUDED_BY_LAYOUT
        elif case == "unoccupied":
            observation = truth.build()
            observation.columns["occupancy_state"][3] = state(Resolution.KNOWN, Basis.DECLARED)
            observation.columns["system_resolution_ptr"] = np.array([0, 1])
            observation.columns["resolution"] = np.array([2.5])
            observation.columns["resolution_kind"] = np.array([ResolutionKind.REFINEMENT], dtype=np.uint8)
            observation.columns["resolution_state"] = np.array(
                [state(Resolution.KNOWN, Basis.DECLARED)], dtype=np.uint8
            )
            truth = Structure.from_arrays(truth.system, observation)
            states[0] = TruthState.OBSERVED_UNOCCUPIED
        expected[states != TruthState.OBSERVED] = 0

        prepared = task(source, cropped=case == "cropped")
        target = prepare_target(prepared, truth, symmetry=POLICY)

        np.testing.assert_array_equal(target.truth, states)
        np.testing.assert_array_equal(target.coordinates, expected)
        assert target.resolution == (2.5 if case == "unoccupied" else None)
        assert not prepared.context.structure.coordinate_mask.any()

    @pytest.mark.parametrize(
        "processor_type",
        [AlphaFold3Processor, ProtenixProcessor, Boltz1Processor, Boltz2Processor, BoltzGenProcessor, ODesignProcessor],
        ids=["af3", "protenix", "boltz1", "boltz2", "boltzgen", "odesign"],
    )
    def test_family_training(self, peptide, processor_type):
        structure = peptide("ALA", "GLY", copies=2)
        coordinates = structure.coordinates + [20, 30, 40]
        truth, _ = replace_coordinates(
            structure,
            np.arange(len(coordinates)),
            coordinates,
            scope=np.ones(len(coordinates), dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        system, observations, _ = project_arrays(
            truth.system.build(), truth.build(), np.array([1, 0]), np.ones(len(coordinates), dtype=bool)
        )
        truth = Structure.from_arrays(MolecularSystem.from_arrays(system), observations)
        processor = processor_type(dtype=torch.float64)
        expected = processor.prepare_input(structure, generator=np.random.default_rng(0))
        prepared, target = processor.prepare(structure, target=truth, generator=np.random.default_rng(0))
        separate = processor.prepare_target(truth, prepared=expected)

        torch.testing.assert_close(prepared.features.to_dict(), expected.features.to_dict())
        torch.testing.assert_close(target.to_dict(), separate.to_dict())
        assert prepared.context.structure is structure
        mask = target.coordinate_mask.reshape(-1, target.coordinate_mask.shape[-1])[0]
        points = target.coordinates.reshape(-1, target.coordinates.shape[-2], 3)[0]
        if processor_type is ODesignProcessor:
            rows = prepared.context.slots.atom[mask.cpu().numpy()]
            expected_points = coordinates[rows].copy()
            fixed = prepared.features.fixed_coordinate_mask[0, mask]
            expected_points[fixed.cpu().numpy()] = prepared.features.observed_positions[0, mask][fixed].cpu().numpy()
        else:
            expected_points = coordinates[mask.cpu().numpy()]
        torch.testing.assert_close(points[mask], torch.from_numpy(expected_points))


class TestComponentAlignment:
    def test_partial_reordered_and_repeated(self, polymers):
        source = Structure.unobserved(polymers(("A", "P", ("ALA", "GLY")), ("B", "Q", ("LYS",))))
        truth = Structure.unobserved(polymers(("B", "Q", ("LYS",)), ("A", "P", ("GLY",))))
        np.testing.assert_array_equal(align_target_components(source, truth, np.array([0, 1, 2, 0])), [1, -1, 0, 1])

    def test_sequence_target_device_preserves_types(self):
        target = SequenceTarget(torch.tensor([3, 7]), torch.tensor([True, False]))
        moved = target.to("cpu")
        torch.testing.assert_close(moved.tokens, target.tokens)
        torch.testing.assert_close(moved.mask, target.mask)
        assert moved.tokens.dtype == torch.int64 and moved.mask.dtype == torch.bool
