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

"""Behavior of ofoldx.models.boltzgen.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data import AtomSite
from ofoldx.data.molecular import ComponentRef, MolecularSystem
from ofoldx.data.projection import project_arrays, project_atoms
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import CODE_NAMES, ContradictionError, InvalidRequestError, UnsupportedError
from ofoldx.models.boltzgen.model import (
    BoltzGenContact,
    BoltzGenDesignMask,
    BoltzGenSecondaryStructure,
)
from ofoldx.models.boltzgen.processor import (
    BOLTZGEN_BACKBONE,
    BoltzGenInverseFoldingProcessor,
    BoltzGenProcessor,
)
from ofoldx.models.conditions import (
    AllowedResidueIdentities,
    GeneratedSegment,
    SymbolicBond,
    SymbolicResidueSite,
    TiedResidueIdentities,
)
from ofoldx.models.design import PositionSelection
from ofoldx.modules.atom.coordinates import gather_observed_coordinates
from ofoldx.msa import MSA
from ofoldx.provenance import Basis, Resolution, state


def feature_name(features) -> str:
    """The atom name one row of ``atom_features`` carries, from its four name-character one-hots."""
    characters = features[132:].reshape(4, 64).argmax(dim=-1).tolist()
    return "".join(chr(code + 32) for code in characters if code)


def slot_names(prepared, rows=None) -> list[str]:
    """The canonical atom names of the prepared model rows (a synthetic row is ``""``)."""
    system = prepared.context.structure.system
    atom = prepared.context.slots.atom if rows is None else prepared.context.slots.atom[rows]
    names = system.text("atom", system.atom_symbol)
    return [str(names[a]) if a >= 0 else "" for a in atom.tolist()]


class TestPlan:
    def test_generated_chain(self, design_library):
        structure, conditions, generated, _ = BoltzGenProcessor().plan(total_len=2, library=design_library)
        system = structure.system
        assert system.text("component", system.component_symbol).tolist() == ["UNK", "UNK"]
        assert generated == tuple(system.component_refs(np.array([0, 1])))
        assert conditions == (BoltzGenDesignMask(generated),)

    def test_generated_chain_appended(self, peptide, design_library):
        placement = {"binder": {"binder": GeneratedSegment("protein", 1, 1)}}
        structure, _, generated, _ = BoltzGenProcessor().plan(
            peptide("ALA", "GLY"), placement=placement, library=design_library
        )
        system = structure.system
        assert system.text("component", system.component_symbol).tolist() == ["ALA", "GLY", "UNK"]
        assert generated == (ComponentRef("chain_0", 0),)
        assert structure.coordinate_mask[: system.component_atom_ptr[2]].all()
        assert not structure.coordinate_mask[system.component_atom_ptr[2] :].any()

    def test_length_budget(self, design_library):
        placement = {"binder": {"binder": GeneratedSegment("protein", 1, 2)}}
        with pytest.raises(InvalidRequestError, match="request_invalid_placement"):
            BoltzGenProcessor().plan(placement=placement, total_len=5, library=design_library)

    def test_nothing_generated(self):
        with pytest.raises(InvalidRequestError, match="request_invalid_placement"):
            BoltzGenProcessor().plan()

    def test_symbolic_condition_resolved(self, design_library):
        placement = {"binder": {"binder": GeneratedSegment("protein", 2, 2)}}
        condition = BoltzGenSecondaryStructure(((SymbolicResidueSite("binder", 1), "H"),))
        _, conditions, _, _ = BoltzGenProcessor().plan(
            placement=placement, conditions=(condition,), library=design_library
        )
        assert conditions[0] == BoltzGenSecondaryStructure(((ComponentRef("chain_0", 1), "H"),))

    def test_link_report_returned(self, peptide):  # CA's HA leaves: its configuration is cleared, and why
        bond = SymbolicBond(AtomSite("A", 1, "CA"), AtomSite("B", 0, "CH3"), leaving_a=frozenset({"HA"}))
        *_, edit = BoltzGenProcessor().plan(peptide("GLY", "ALA", "GLY", ligands=("ACT",)), conditions=(bond,))
        assert [CODE_NAMES[r] for r in edit.parity_clears.reason] == ["parity_cleared_heavy_neighbour_removed"]


class TestPrepare:
    @pytest.fixture
    def prepared(self, peptide):
        conditions = (BoltzGenDesignMask((ComponentRef("A", 0),)),)
        return BoltzGenProcessor(dtype=torch.float64).prepare_input(peptide("ALA", "GLY", "LYS"), conditions=conditions)

    def test_design_scope(self, prepared):
        system = prepared.context.structure.system
        owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
        assert prepared.context.scope.rule == "components" and prepared.context.scope.frame == "source"
        assert np.unique(owner[prepared.context.scope.atoms]).tolist() == [0]

    def test_designed_atom14_rows(self, prepared):
        names = slot_names(prepared, np.arange(14))
        assert names == ["N", "CA", "C", "O", "CB"] + [""] * 9
        assert prepared.features.atom_index[0, :14].tolist() == prepared.context.slots.atom[:14].tolist()

    def test_designed_side_chain_flagged(self, prepared):
        features = prepared.features.atom_features[0]
        element = features[:14, 4:132].argmax(dim=-1)
        assert element[:4].tolist() == [7, 6, 6, 8]
        assert element[4:].eq(114).all() and features[4:14, 3].eq(0).all()
        assert feature_name(features[4]) == "FL4"

    def test_designed_identity_masked(self, prepared):
        tokens = prepared.features.residue_type_features[0].argmax(dim=-1).tolist()
        assert tokens == [TOKEN_INDEX[token] for token in "XGK"]
        assert prepared.features.design_mask[0].tolist() == [True, False, False]
        assert prepared.features.msa_tokens[0, 0].tolist() == [TOKEN_INDEX[token] for token in "XGK"]
        assert prepared.features.msa_profile[0, 0].argmax().item() == TOKEN_INDEX["X"]

    def test_connected_design_scope(self, prepared):
        assert prepared.features.chain_design_mask[0].tolist() == [True, True, True]

    def test_contact_channels(self, peptide):
        contact = BoltzGenContact((ComponentRef("A", 0),), (ComponentRef("A", 2),), 8.0, "binder>pocket")
        prepared = BoltzGenProcessor().prepare_input(peptide("ALA", "GLY", "LYS"), conditions=(contact,))
        classes = prepared.features.contact_conditioning[0].argmax(dim=-1)
        assert classes[0, 2].item() == 3 and classes[2, 0].item() == 2
        assert classes[0, 1].item() == 1
        assert prepared.features.contact_threshold[0, 0, 2].item() == 8.0

    def test_unsupported_identity(self, peptide):
        conditions = (BoltzGenDesignMask((ComponentRef("A", 1),)),)
        with pytest.raises(UnsupportedError, match="model_capacity"):
            BoltzGenProcessor().prepare_input(peptide("ALA", "MSE"), conditions=conditions)

    def test_overfull_link_refused(self, peptide):  # the link keeps every count: CH3 bonded to SG is five-valent
        bond = SymbolicBond(AtomSite("A", 1, "SG"), AtomSite("B", 0, "CH3"), leaving_a=frozenset({"HG"}))
        processor = BoltzGenProcessor()
        structure, conditions, _, _ = processor.plan(peptide("GLY", "CYS", "GLY", ligands=("ACT",)), conditions=(bond,))
        with pytest.raises(ContradictionError, match="prepare_valence_excess"):
            processor.prepare_input(structure, conditions=conditions)

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            BoltzGenProcessor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = BoltzGenProcessor(dtype=torch.float64)
        structure = peptide("ALA", "GLY")
        prepared, truth = processor.prepare(structure, target=structure, generator=np.random.default_rng(5))
        tensors = truth
        assert tensors.coordinates.shape == (1, 1, len(structure.coordinates), 3)
        assert tensors.coordinate_atom_index.shape == (1, len(prepared.context.slots.atom))
        assert tensors.residue_type.shape == (1, 2, len(TOKENS))
        assert tensors.chain_span.tolist() == [[[0, len(structure.coordinates)]]]

    @pytest.mark.parametrize("input_known", [True, False], ids=["known-input", "unknown-input"])
    def test_truth_b_factors(self, peptide, input_known):
        original = peptide("ALA", "GLY", copies=2)
        arrays = original.build()
        arrays.columns["b_factor"][:] = 10 if input_known else 0
        arrays.columns["b_factor_state"][:] = state(
            Resolution.KNOWN if input_known else Resolution.UNKNOWN, Basis.DECLARED
        )
        structure = Structure.from_arrays(original.system, arrays)
        arrays = original.build()
        sizes = np.diff(original.system.component_atom_ptr)
        arrays.columns["b_factor"][:] = np.repeat([60, 0, 80, 90], sizes)
        arrays.columns["b_factor_state"][:] = np.repeat(
            [state(Resolution.KNOWN, Basis.DECLARED), state(Resolution.UNKNOWN, Basis.DECLARED)]
            + [state(Resolution.KNOWN, Basis.DECLARED)] * 2,
            sizes,
        )
        unoccupied, missing = original.system.atom_index([AtomSite("A", 0, "CA"), AtomSite("B", 1, "CA")])
        arrays.columns["occupancy"][unoccupied] = 0
        arrays.columns["occupancy_state"][unoccupied] = state(Resolution.KNOWN, Basis.DECLARED)
        target = Structure.from_arrays(original.system, arrays)
        keep = np.arange(len(target.coordinates)) != missing
        target, _ = project_atoms(target, keep)
        system, observations, _ = project_arrays(
            target.system.build(), target.build(), np.array([1, 0]), np.ones(len(target.coordinates), dtype=bool)
        )
        assert observations is not None
        target = Structure.from_arrays(MolecularSystem.from_arrays(system), observations)
        processor = BoltzGenProcessor()
        prepared, truth = processor.prepare(structure, target=target, generator=np.random.default_rng(0))
        tensors = truth

        assert tensors.b_factor.tolist() == [[60, 0, 80, 0]]
        assert tensors.b_factor_mask.tolist() == [[True, False, True, False]]
        assert not tensors.coordinate_mask[0, unoccupied]


class TestUnknownAtom14:
    @pytest.mark.parametrize("cb", ("observed", "unobserved", "absent"))
    def test_unknown_cb_and_nine_nitrogen_markers(self, design_library, cb):
        processor = BoltzGenProcessor()
        if cb == "absent":
            structure, conditions, _, _ = processor.plan(
                None,
                placement={"binder": {"binder": GeneratedSegment("protein", 1, 1)}},
                library=design_library,
            )
            instance = "chain_0"
        else:
            system, _, _ = build_system(
                Recipe({"P": PolymerSequence("polypeptide(L)", ("UNK",))}, {"A": "P"}), design_library
            )
            structure = Structure.unobserved(system)
            conditions = (BoltzGenDesignMask((ComponentRef("A", 0),)),)
            instance = "A"
        atom_names = ["N", "CA", "C", "O"] + (["CB"] if cb == "observed" else [])
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], [1, -1, 0]], dtype=float)
        indexes = structure.system.atom_index([AtomSite(instance, 0, name) for name in atom_names])
        structure, _ = replace_coordinates(
            structure,
            indexes,
            positions[: len(indexes)],
            scope=np.ones(len(structure.system.atom_symbol), dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        prepared, target = processor.prepare(structure, target=structure, conditions=conditions)
        cb_index = int(structure.system.atom_index([AtomSite(instance, 0, "CB")])[0])
        assert prepared.features.atom_index[0, 4].item() == cb_index
        assert prepared.features.atom_coordinate_mask[0, 4].item() is (cb == "observed")
        assert prepared.features.atom_coordinate_mask[0, 5:14].all()
        values = prepared.features.atom_coordinates[0, :14]
        torch.testing.assert_close(values[5:14], values[0].expand(9, 3))
        if cb == "observed":
            torch.testing.assert_close(values[4] - values[0], torch.tensor([1.0, -1.0, 0.0]))
        else:
            torch.testing.assert_close(values[4], torch.zeros(3))
        assert target.coordinate_atom_index[0, 4].item() == cb_index
        gathered, observed = gather_observed_coordinates(
            target.coordinates[:, 0],
            target.coordinate_mask,
            target.coordinate_atom_index,
            prepared.features.atom_mask,
        )
        assert observed[0, 4].item() is (cb == "observed")
        assert observed[0, 5:14].all()
        torch.testing.assert_close(gathered[0, 5:14], gathered[0, 0].expand(9, 3))
        assert feature_name(prepared.features.atom_features[0, 4]) == "FL4"


class TestInverseFolding:
    @pytest.fixture
    def prepared(self, peptide):
        return BoltzGenInverseFoldingProcessor(dtype=torch.float64).prepare_input(peptide("ALA", "GLY", "LYS"))

    def test_sequence_scope(self, prepared):
        assert prepared.context.scope.rule == "none" and not prepared.context.scope.atoms.any()
        assert prepared.features.design_mask[0].tolist() == [True, True, True]

    def test_backbone_addressed(self, prepared):
        names = slot_names(prepared)
        assert len(names) == 12 and set(names) == set(BOLTZGEN_BACKBONE)

    @pytest.mark.parametrize("with_msa", [False, True])
    def test_designed_msa_profile_is_masked(self, peptide, with_msa):
        structure = peptide("ALA", "GLY", "LYS")
        processor = BoltzGenInverseFoldingProcessor()
        msa = None
        if with_msa:
            counts = np.array([[0, 0, 0], [2, 3, 4]], dtype=np.uint32)
            msa = {"A": MSA("P", ("AGK", "AKK"), counts, ("AGK",), counts[:1])}
        baseline = processor.prepare_input(structure, design=PositionSelection(components=()), msa=msa).features
        prepared = processor.prepare_input(
            structure, design=PositionSelection(components=(ComponentRef("A", 1),)), msa=msa
        ).features
        assert prepared.design_mask[0].tolist() == [False, True, False]
        assert prepared.residue_type_features[0, 1].argmax().item() == TOKEN_INDEX["X"]
        assert prepared.msa_profile[0, 1].argmax().item() == TOKEN_INDEX["X"]
        assert prepared.msa_profile[0, 1, TOKEN_INDEX["X"]].item() == 1
        assert prepared.deletion_mean[0, 1].item() == 0
        torch.testing.assert_close(prepared.msa_profile[0, [0, 2]], baseline.msa_profile[0, [0, 2]])
        torch.testing.assert_close(prepared.deletion_mean[0, [0, 2]], baseline.deletion_mean[0, [0, 2]])

    def test_omitted_identity(self, peptide):
        prepared = BoltzGenInverseFoldingProcessor().prepare_input(peptide("ALA", "GLY"), global_omit_tokens=("C",))
        allowed = prepared.features.allowed_token_mask[0]
        assert not allowed[:, TOKEN_INDEX["C"]].any()
        assert allowed[:, TOKEN_INDEX["A"]].all()
        assert allowed.sum(dim=-1).tolist() == [19, 19]

    def test_allowed_identities(self, peptide):
        condition = AllowedResidueIdentities((ComponentRef("A", 0),), frozenset({"ALA"}))
        prepared = BoltzGenInverseFoldingProcessor().prepare_input(peptide("ALA", "GLY"), conditions=(condition,))
        allowed = prepared.features.allowed_token_mask[0]
        assert allowed[0].sum().item() == 1 and allowed[0, TOKEN_INDEX["A"]]
        assert allowed[1].sum().item() == 20

    def test_tied_identities(self, peptide):
        groups = ((ComponentRef("A", 0), ComponentRef("A", 2)),)
        condition = TiedResidueIdentities(groups)
        prepared = BoltzGenInverseFoldingProcessor().prepare_input(
            peptide("ALA", "GLY", "LYS"), conditions=(condition,)
        )
        assert prepared.features.symmetric_group[0].tolist() == [1, 0, 1]

    def test_overlapping_tied_groups(self, peptide):
        groups = ((ComponentRef("A", 0), ComponentRef("A", 1)), (ComponentRef("A", 1), ComponentRef("A", 2)))
        condition = TiedResidueIdentities(groups)
        prepared = BoltzGenInverseFoldingProcessor().prepare_input(
            peptide("ALA", "GLY", "LYS"), conditions=(condition,)
        )
        assert prepared.features.symmetric_group[0].tolist() == [1, 1, 1]

    def test_sequence_target(self, peptide):
        processor = BoltzGenInverseFoldingProcessor(dtype=torch.float64)
        prepared, target = processor.prepare(peptide("ALA", "GLY", "LYS"), target=peptide("ALA", "GLY", "LYS"))
        tensors = target
        assert tensors.tokens.shape == (1, 3)
        assert tensors.mask.all()
        assert tensors.tokens[0].tolist() == [TOKEN_INDEX[token] for token in "AGK"]


class TestInverseFoldingTargetAlignment:
    def test_reordered_truth_and_missing_components(self, peptide, polymers):
        structure = peptide("ALA", "GLY", copies=2)
        processor = BoltzGenInverseFoldingProcessor()
        prepared = processor.prepare_input(structure)
        truth = Structure.unobserved(polymers(("B", "Q", ("LYS", "ALA")), ("A", "P", ("GLY", "LEU"))))
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.tokens.tolist() == [[TOKEN_INDEX[t] for t in "GLKA"]]
        assert target.mask.all()
        partial = Structure.unobserved(polymers(("A", "P", ("GLY", "LEU"))))
        labels = processor.prepare_target(partial, prepared=prepared)
        assert labels.mask.tolist() == [[True, True, False, False]]
        assert labels.tokens[0, 2:].tolist() == [TOKEN_INDEX["<unk>"]] * 2

    def test_structure_target_requires_component_identity(self, peptide, polymers):
        structure = peptide("ALA", "GLY")
        processor = BoltzGenProcessor()
        prepared = processor.prepare_input(structure)
        missing = Structure.unobserved(polymers(("A", "P", ("ALA",))))
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            processor.prepare_target(missing, prepared=prepared)
