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

"""Behavior of ofoldx.models.nampnn.processor."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ofoldx.models.nampnn import NaMpnnProcessor
from ofoldx.models.nampnn.config import NaMpnnProcessorConfig


def classes(*tokens: str) -> list[list[int]]:
    return [[TOKEN_INDEX[token] for token in tokens]]


class TestPrepare:
    @pytest.mark.parametrize("shared,rna", [(True, "DNA_A"), (False, "RNA_A")])
    def test_row_classes(self, complex_structure, shared, rna):
        processor = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=shared))
        prepared = processor.prepare_input(complex_structure)
        assert prepared.features.sequence_tokens.tolist() == classes(rna, rna, "A", "G")
        assert prepared.features.polymer_type_index.tolist() == [[2, 2, 0, 0]]
        assert prepared.context.scope.rule == "none"

    def test_nucleic_acid_only(self, complex_structure):
        prepared = NaMpnnProcessor().prepare_input(complex_structure, nucleic_acid_only=True)
        assert prepared.features.design_mask.tolist() == [[True, True, False, False]]

    def test_allowed_identities(self, complex_structure):
        condition = AllowedResidueIdentities((ComponentRef("A", 0),), frozenset({"G"}))
        processor = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False))
        allowed = processor.prepare_input(complex_structure, conditions=(condition,)).features.allowed_token_mask[0, 0]
        assert allowed.nonzero().flatten().tolist() == [TOKEN_INDEX["RNA_G"]]

    def test_tie_across_types(self, complex_structure):
        tie = TiedResidueIdentities(((ComponentRef("A", 0), ComponentRef("B", 0)),))
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            NaMpnnProcessor().prepare_input(complex_structure, conditions=(tie,))

    def test_untrained_class_reads_unknown(self, peptide):
        prepared = NaMpnnProcessor().prepare_input(peptide("ASX", "ALA"))
        assert prepared.features.sequence_tokens.tolist() == classes("X", "A")
        assert prepared.context.record.records.counts()["encoding_loss"] == 1

    def test_shared_rna_omit_refused(self, complex_structure):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            NaMpnnProcessor().prepare_input(complex_structure, global_omit_tokens=("RNA_U",))

    @pytest.mark.parametrize("sequence", ["KB", "K"])
    def test_sequence_refused(self, complex_structure, sequence):
        with pytest.raises(InvalidRequestError, match="request_invalid_sequence"):
            NaMpnnProcessor().prepare_input(complex_structure, sequence={"B": sequence})

    def test_sequence_override(self, complex_structure):
        prepared = NaMpnnProcessor().prepare_input(complex_structure, sequence={"B": "KA"})
        assert prepared.features.sequence_tokens[0, 2:].tolist() == [TOKEN_INDEX["K"], TOKEN_INDEX["A"]]


class TestSequenceTarget:
    @pytest.mark.parametrize("shared", [False, True])
    def test_truth_respects_declared_nucleic_columns(self, complex_structure, polymers, shared):
        source = complex_structure
        truth = Structure.unobserved(polymers(("A", "R", ("G", "U")), ("B", "P", ("GLY", "ALA"))))
        processor = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=shared))
        baseline = processor.prepare_input(source)
        prepared, target = processor.prepare(source, target=truth)
        prefix = "DNA_" if shared else "RNA_"
        assert target.tokens.tolist() == [
            [TOKEN_INDEX[prefix + ("T" if shared and token == "U" else token)] for token in "GU"]
            + [TOKEN_INDEX["G"], TOKEN_INDEX["A"]]
        ]
        assert target.mask.all()
        torch.testing.assert_close(prepared.features.to_dict(), baseline.features.to_dict())

    def test_protein_truth_is_independent(self, peptide):
        source, truth = peptide("ALA", "GLY"), peptide("GLY", "ALA")
        processor = NaMpnnProcessor()
        prepared, target = processor.prepare(source, target=truth)
        assert target.tokens.tolist() == [[TOKEN_INDEX["G"], TOKEN_INDEX["A"]]]
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX["A"], TOKEN_INDEX["G"]]]


class TestPreparedRepresentation:
    def test_different_nucleic_columns_reject_cached_input(self, complex_structure):
        source = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False))
        other = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=True))
        prepared = source.prepare_input(complex_structure)
        with pytest.raises(InvalidRequestError):
            other.prepare_target(complex_structure, prepared=prepared)

    def test_equivalent_processor_reuses_context_across_dtype(self, complex_structure):
        first = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False), dtype=torch.float32)
        second = NaMpnnProcessor(NaMpnnProcessorConfig(shared_nucleic_tokens=False), dtype=torch.float64)
        prepared = first.prepare_input(complex_structure)
        expected = first.prepare_target(complex_structure, prepared=prepared)
        actual = second.prepare_target(complex_structure, prepared=prepared)
        torch.testing.assert_close(actual.tokens, expected.tokens)
        torch.testing.assert_close(actual.mask, expected.mask)
