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

"""Behavior of ofoldx.models.alphafold2.processor."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data import residue_constants
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import UnsupportedError
from ofoldx.models.alphafold2 import AlphaFold2InitialGuessProcessor, AlphaFold2MultimerProcessor, AlphaFold2Processor
from ofoldx.msa import MSA
from ofoldx.templates import ChainCoordinates, PolymerCoordinates, Template, TemplateAlignment

_CA = residue_constants.atom_order["CA"]
_TYPES = residue_constants.AMINO_ACID_ONE_LETTER_CODES


@pytest.fixture
def glycine_template() -> Template:
    """A one-residue GLY template chain ``T`` (N, CA, C placed) aligned to position 1 of entity ``P``."""
    chain = ChainCoordinates(
        polymer_type="polypeptide(L)",
        component_ids=("GLY",),
        token_class=np.array([TOKEN_INDEX["G"]], dtype=np.uint8),
        encoding=np.zeros(1, dtype=np.uint8),
        atom_name=np.array([0, 1, 2], dtype=np.int32),
        atom_symbols=("N", "CA", "C"),
        component_index=np.zeros(3, dtype=np.int64),
        coordinates=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.4, 0.0]]),
        coordinate_mask=np.ones(3, dtype=bool),
        parent_component_ids=((),),
    )
    return Template(PolymerCoordinates({"T": chain}), (TemplateAlignment("P", None, "T", ((1, 0),)),))


class TestPrepare:
    def test_atom37_slots(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = AlphaFold2Processor().prepare_input(structure)
        atoms = prepared.context.slots.atom.reshape(-1, residue_constants.NUM_ATOM_TYPES)
        names = structure.system.text("atom", structure.system.atom_symbol[atoms[:, _CA]])
        assert names.tolist() == ["CA", "CA"]
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert prepared.features.token_features.shape == (1, 2, 1 + len(TOKEN_INDEX))

    def test_mse_reads_methionine(self, peptide):
        prepared = AlphaFold2Processor().prepare_input(peptide("ALA", "MSE"))
        assert prepared.features.residue_type_ids[0].tolist() == [_TYPES.index("A"), _TYPES.index("M")]

    def test_ligand_not_addressed(self, peptide):
        prepared = AlphaFold2Processor().prepare_input(peptide("ALA", ligands=("ACT",)))
        assert prepared.features.residue_type_ids.shape == (1, 1)

    def test_two_chains_refused(self, chains):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            AlphaFold2Processor().prepare_input(chains(A=("ALA",), B=("GLY",)))

    def test_msa_pool(self, peptide):
        msa = MSA("P", ("AG", "AA", "AG"), np.zeros((3, 2), np.uint32), ("AG",), np.zeros((1, 2), np.uint32))
        prepared = AlphaFold2Processor().prepare_input(peptide("ALA", "GLY"), msa={"A": msa})
        tokens = prepared.features.msa_tokens[0]
        assert tokens[:, 1].tolist() == [TOKEN_INDEX["G"], TOKEN_INDEX["A"]]  # the duplicate row is dropped
        assert prepared.features.msa_profile.shape == (1, 2, len(TOKEN_INDEX))

    def test_template_features(self, peptide, glycine_template):
        prepared = AlphaFold2Processor().prepare_input(peptide("ALA", "GLY"), templates=(glycine_template,))
        gap = residue_constants.NUM_AMINO_ACID_TYPES + 1
        assert prepared.features.template_residue_type_ids[0, 0].tolist() == [gap, _TYPES.index("G")]
        assert prepared.features.template_all_atom_mask[0, 0, :, _CA].tolist() == [0.0, 1.0]


class TestInitialGuess:
    def test_binder_roles(self, chains):
        structure = chains(B=("GLY",), A=("ALA", "GLY"))
        prepared = AlphaFold2InitialGuessProcessor().prepare_input(structure, binder_instance_id="B")
        assert prepared.features.binder_mask[0].tolist() == [True, False, False]
        assert prepared.features.residue_index[0].tolist() == [51, 0, 1]  # the binder follows the target chain


class TestMultimer:
    def test_chain_features(self, chains):
        prepared = AlphaFold2MultimerProcessor().prepare_input(chains(A=("ALA",), B=("GLY",), C=("ALA",)))
        assert prepared.features.chain_index[0].tolist() == [1, 2, 3]
        assert prepared.features.entity_index[0].tolist() == [1, 2, 1]
        assert prepared.features.symmetry_index[0].tolist() == [1, 1, 2]
