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

"""Behavior of ofoldx.models.proteinmpnn.processor."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef, MolecularSystem, SystemArrays
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.design import PositionSelection
from ofoldx.models.proteinmpnn import ProteinMpnnProcessor


class TestPrepare:
    def test_protein_rows(self, peptide):
        prepared = ProteinMpnnProcessor().prepare_input(peptide("ALA", "MSE", "GLY", ligands=("ACT",)))
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX[token] for token in "AMG"]]
        assert prepared.features.design_mask.all()
        assert prepared.context.scope.rule == "none"
        assert prepared.context.record.records.counts()["encoding_loss"] == 1

    def test_cap_is_no_row(self, peptide):
        prepared = ProteinMpnnProcessor().prepare_input(peptide("ACE", "ALA", "GLY"))
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX["A"], TOKEN_INDEX["G"]]]

    def test_no_protein_no_rows(self, peptide):
        prepared = ProteinMpnnProcessor().prepare_input(peptide("ACE"))
        assert prepared.features.sequence_tokens.shape == (1, 0)

    def test_design_selection(self, peptide):
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        prepared = ProteinMpnnProcessor().prepare_input(peptide("ALA", "GLY"), design=selection)
        assert prepared.features.design_mask.tolist() == [[False, True]]

    def test_allowed_identities(self, peptide):
        condition = AllowedResidueIdentities((ComponentRef("A", 0),), frozenset({"LYS"}))
        prepared = ProteinMpnnProcessor().prepare_input(peptide("ALA", "GLY"), conditions=(condition,))
        allowed = prepared.features.allowed_token_mask[0, 0]
        assert allowed.nonzero().flatten().tolist() == [TOKEN_INDEX["K"]]

    @pytest.mark.parametrize("sequence,valid", [("GA", True), ("G", False), ("GB", False)])
    def test_sequence_override(self, peptide, sequence, valid):
        processor = ProteinMpnnProcessor()
        if not valid:
            with pytest.raises(InvalidRequestError, match="request_invalid_sequence"):
                processor.prepare_input(peptide("ALA", "GLY"), sequence=sequence)
            return
        prepared = processor.prepare_input(peptide("ALA", "GLY"), sequence=sequence)
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX["G"], TOKEN_INDEX["A"]]]


class TestSequenceTarget:
    def test_prepare_pair_and_separate_target_agree(self, peptide):
        source = peptide("ALA", "GLY", "LYS")
        truth = peptide("GLY", "ALA", "SER")
        processor = ProteinMpnnProcessor()
        baseline = processor.prepare_input(source)
        prepared, target = processor.prepare(source, target=truth)
        separate = processor.prepare_target(truth, prepared=prepared)
        torch.testing.assert_close(prepared.features.to_dict(), baseline.features.to_dict())
        assert prepared.context.structure is source
        assert target.tokens.tolist() == [[TOKEN_INDEX[token] for token in "GAS"]]
        assert target.mask.all()
        torch.testing.assert_close(target.tokens, separate.tokens)
        torch.testing.assert_close(target.mask, separate.mask)
        unlabelled, absent = processor.prepare(source)
        assert absent is None
        torch.testing.assert_close(unlabelled.features.to_dict(), baseline.features.to_dict())

    def test_partial_truth_is_not_filled_from_source(self, peptide):
        source = peptide("ALA", "GLY", "LYS")
        truth = peptide("GLY", "ALA")
        processor = ProteinMpnnProcessor()
        prepared = processor.prepare_input(source)
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.mask.tolist() == [[True, True, False]]
        assert target.tokens[0, 2] == TOKEN_INDEX["<unk>"]
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX[token] for token in "AGK"]]

    def test_known_untrained_classes_remain_known(self, peptide, polymers):
        source = peptide("ALA", "GLY")
        arrays = polymers(("A", "P", ("ALA", "GLY"))).build()
        # Sequence truth can be known even when only partial chemical observations exist.
        symbols = {**arrays.symbols, "component": ("SEC", "PYL")}
        truth = Structure.unobserved(MolecularSystem.from_arrays(SystemArrays(symbols, arrays.columns)))
        processor = ProteinMpnnProcessor()
        prepared = processor.prepare_input(source)
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.tokens.tolist() == [[TOKEN_INDEX["U"], TOKEN_INDEX["O"]]]
        assert target.mask.all()
