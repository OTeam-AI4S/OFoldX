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

"""Behavior of ofoldx.models.esmfold2.processor."""

from __future__ import annotations

import pytest

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import UnsupportedError
from ofoldx.models.esmfold2 import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    is_antibody_sequence,
)


@pytest.mark.parametrize("operation", [is_antibody_sequence, antibody_cdr_indices])
def test_antibody_numbering_reports_missing_hmmer(operation, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="HMMER 3.*hmmscan.*PATH"):
        operation("AG")


class TestPrepare:
    def test_prediction_scope(self, peptide):
        prepared = EsmFold2Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert prepared.features.residue_type_ids.shape[-1] == len(prepared.context.slots.layout.token_component)
        assert prepared.features.molecule_type[0].tolist() == [0, 0, 3, 3, 3, 3]
        assert prepared.context.record.records.counts()["encoding_loss"] == 1

    def test_designated_leaving_unaddressed(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = EsmFold2Processor().prepare_input(structure)
        system = structure.system
        assert "OXT" not in system.text("atom", system.atom_symbol[prepared.context.slots.atom]).tolist()

    def test_representative_atom(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = EsmFold2Processor().prepare_input(structure)
        system = structure.system
        atoms = prepared.context.slots.atom[prepared.features.token_representative_atom_index[0].numpy()]
        assert system.text("atom", system.atom_symbol[atoms]).tolist() == ["CB", "CA"]

    def test_language_model_rows(self, peptide):
        prepared = EsmFold2Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert prepared.features.token_has_esmc[0].tolist() == [True, True, False, False, False, False]
        assert prepared.features.language_model_tokens.shape[0] == 1
        assert prepared.features.token_to_esmc[0, :2].tolist() == [1, 2]

    def test_token_bonds(self, peptide):
        prepared = EsmFold2Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        bonds = prepared.features.token_bonds[0, ..., 0]
        assert bonds[0, 1] == 1 and bonds[1, 0] == 1  # the polymer junction, unlike Boltz-1
        assert bonds[2:, 2:].any()

    def test_msa_rows(self, peptide, msa_pool):
        prepared = EsmFold2Processor().prepare_input(peptide("ALA", "GLY"), msa=msa_pool("AG", "AA"))
        assert prepared.features.msa_tokens[0, 0].tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["G"]]
        assert prepared.features.msa_tokens[0, 1].tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["A"]]
        assert prepared.features.msa_mask.all()

    def test_fast_variant_refuses_msa(self, peptide, msa_pool):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            EsmFold2ExperimentalProcessor().prepare_input(peptide("ALA", "GLY"), msa=msa_pool("AG", "AA"))
