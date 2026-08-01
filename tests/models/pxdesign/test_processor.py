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

"""Behavior of ofoldx.models.pxdesign.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data import project_atoms
from ofoldx.data.molecular import ComponentRef
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError, RequestRef
from ofoldx.models.conditions import ExistingSpan, GeneratedSegment, Hotspot
from ofoldx.models.pxdesign.processor import GENERATED_ATOMS, PxDesignProcessor, from_binder, to_binder


def generated_slots(prepared) -> np.ndarray:
    """The slots whose atom the model generates."""
    return np.isin(prepared.context.slots.atom, np.flatnonzero(prepared.context.scope.atoms))


def slot_names(prepared, rows) -> list[str]:
    system = prepared.context.structure.system
    return system.text("atom", system.atom_symbol[prepared.context.slots.atom[rows]]).tolist()


class TestBinderPlacement:
    def test_round_trip(self):
        placement = from_binder(3, [("A", [(0, 2), (4, 6)])])
        assert to_binder(placement) == (3, [("A", [(0, 2), (4, 6)])])

    @pytest.mark.parametrize("length,target", [(0, []), (-1, []), (2, [("A", [])]), (2, [("A", [(5, 5)])])])
    def test_invalid_request(self, length, target):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            from_binder(length, target)

    def test_binder_required(self):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            to_binder({"A": {"A": ExistingSpan("A", 0, 1)}})

    def test_one_fixed_length(self):
        placement = {"binder": {"binder": GeneratedSegment("protein", 1, 3)}}
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            to_binder(placement)


class TestPrepare:
    @pytest.fixture
    def prepared(self, peptide, design_library):
        structure = peptide("ALA", "GLY", "LYS")
        placement = from_binder(2, [("A", [(0, 2)])])
        processor = PxDesignProcessor(dtype=torch.float64)
        return processor.prepare_input(structure, placement=placement, library=design_library)

    @pytest.fixture
    def cropped(self, peptide):
        """ALA-GLY-LYS holding ALA's atoms only."""
        structure = peptide("ALA", "GLY", "LYS")
        owner = np.repeat(np.arange(3), np.diff(structure.system.component_atom_ptr))
        return project_atoms(structure, owner == 0)[0]

    def test_generated_scope(self, prepared):
        assert prepared.context.scope.rule == "components" and prepared.context.scope.frame == "source"
        system = prepared.context.structure.system
        owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
        scoped = np.unique(owner[prepared.context.scope.atoms])
        assert scoped.tolist() == [3, 4]  # the two generated residues, after the target's three components

    def test_target_span_cropped(self, prepared):
        # omission is not deletion: the cropped-away residue keeps its place in its instance, without an atom
        system = prepared.context.structure.system
        ids = system.text("component", system.component_symbol).tolist()
        assert ids == ["ALA", "GLY", "LYS", "UNK", "UNK"]
        assert np.diff(system.component_atom_ptr).tolist()[2] == 0

    def test_generated_tokens_unknown(self, prepared):
        tokens = prepared.features.residue_type_features[0].argmax(dim=-1).tolist()
        design = prepared.features.design_features[0, :, 0].tolist()
        assert tokens == [TOKEN_INDEX[token] for token in "AGXX"]
        assert design == [0.0, 0.0, 1.0, 1.0]

    def test_generated_backbone_only(self, prepared):
        rows = generated_slots(prepared)
        assert slot_names(prepared, rows) == list(GENERATED_ATOMS) * 2

    def test_generated_backbone_template(self, prepared, backbone_template):
        rows = generated_slots(prepared)
        positions = prepared.features.reference_positions[0].numpy()[rows]
        np.testing.assert_allclose(positions, np.concatenate((backbone_template, backbone_template)), atol=1e-6)
        assert prepared.context.record.records.counts()["design_reference_substituted"] == 2

    def test_self_template_masked(self, prepared):
        mask = prepared.features.template_pair_mask[0]
        assert mask[:2, :2].all() and not mask[2:].any() and not mask[:, 2:].any()

    def test_hotspot_tokens(self, peptide, design_library):
        placement = from_binder(1, [("A", [(0, 2)])])
        condition = Hotspot((ComponentRef("A", 1),))
        prepared = PxDesignProcessor().prepare_input(
            peptide("ALA", "GLY"), placement=placement, conditions=(condition,), library=design_library
        )
        assert prepared.features.hotspot[0, :, 0].tolist() == [0.0, 1.0, 0.0]

    def test_target_span_without_atoms(self, cropped, design_library):  # beside a span holding atoms
        placement = from_binder(2, [("A", [(0, 1), (1, 3)])])
        prepared = PxDesignProcessor().prepare_input(cropped, placement=placement, library=design_library)
        assert prepared.features.design_features[0, :, 0].tolist() == [0.0, 1.0, 1.0]

    def test_target_without_atoms_refused(self, cropped, design_library):  # conditioning on nothing
        placement = from_binder(2, [("A", [(1, 3)])])
        with pytest.raises(InvalidRequestError, match="request_invalid_positions") as raised:
            PxDesignProcessor().prepare_input(cropped, placement=placement, library=design_library)
        assert raised.value.refs == (RequestRef("placement", "A[1, 3)"),)

    @pytest.mark.parametrize("target", [[("A", [(0, 1)])], []])
    def test_hotspot_outside_spans(self, peptide, design_library, target):
        placement = from_binder(1, target)
        condition = Hotspot((ComponentRef("A", 1),))
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            PxDesignProcessor().prepare_input(
                peptide("ALA", "GLY"), placement=placement, conditions=(condition,), library=design_library
            )

    def test_binder_without_target(self, design_library):
        placement = from_binder(2)
        prepared = PxDesignProcessor().prepare_input(placement=placement, library=design_library)
        assert prepared.features.design_features[0, :, 0].tolist() == [1.0, 1.0]
        assert not prepared.features.template_pair_mask.any()
