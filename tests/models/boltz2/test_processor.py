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

"""Behavior of ofoldx.models.boltz2.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.boltz2 import Boltz2Contact, Boltz2Template
from ofoldx.models.boltz2.processor import (
    Boltz2AffinityProcessor,
    Boltz2Processor,
    standardize_boltz2_affinity_smiles,
)
from ofoldx.msa import MSA


class TestPrepare:
    def test_prediction_scope(self, peptide):
        prepared = Boltz2Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert prepared.features.residue_type_features.shape[-1] == len(TOKENS)

    def test_modified_residue_token(self, peptide):
        input = Boltz2Processor().prepare_input(peptide("ALA", "MSE", "GLY")).features
        assert input.modified[0].tolist() == [0, 1, 0]
        assert input.residue_type_features[0].argmax(-1).tolist() == [TOKEN_INDEX[token] for token in "AMG"]

    def test_token_bonds(self, peptide):
        input = Boltz2Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",))).features
        bonds, types = input.token_bonds[0], input.type_bonds[0]
        assert bonds[:2].sum() == 0  # no polymer junction, as upstream
        assert bonds[2:, 2:].any() and set(types[bonds.bool()].tolist()) <= {2, 3, 5}

    def test_unknown_method_refused(self, peptide):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            Boltz2Processor().prepare_input(peptide("ALA"), method="cryo")

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            Boltz2Processor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = Boltz2Processor(dtype=torch.float64)
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure, generator=np.random.default_rng(3))
        tensors = target
        assert tensors.coordinates.shape == (1, 1, len(structure.coordinates), 3)
        assert tensors.chain_span.tolist() == [[[0, len(structure.coordinates)]]]


class TestTemplates:
    def test_one_row_per_template(self, peptide, template):
        structure = peptide("ALA", "GLY")
        value = template(structure, "A", (None, ((0, 0), (1, 1))))
        input = (
            Boltz2Processor().prepare_input(structure, templates=(value, Boltz2Template(value, force=True))).features
        )
        assert input.template_mask.shape == (1, 2, 2)
        assert input.template_force[0].tolist() == [False, True]
        assert input.template_frame_mask[0].bool().all()

    def test_reused_chain_frames(self, peptide, template):
        structure = peptide("ALA", "GLY", copies=2)
        value = template(structure, "A", (None, ((0, 0), (1, 1))))
        frames = Boltz2Processor().prepare_input(structure, templates=(value,)).features.template_visibility_index[0, 0]
        assert frames[0] == frames[1] and frames[2] == frames[3] and frames[0] != frames[2]

    def test_untemplated_tokens_join_chain(self, peptide, template):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        value = template(structure, "A", ("A", ((0, 0),)))
        input = Boltz2Processor().prepare_input(structure, templates=(value,)).features
        frames = input.template_visibility_index[0, 0]
        assert frames[0] == frames[1] and (frames[2:] == frames[2]).all() and frames[2] != frames[0]
        assert input.template_residue_type_features[0, 0, 1].argmax() == TOKEN_INDEX["<pad>"]

    def test_self_template_row(self, peptide):
        input = Boltz2Processor().prepare_input(peptide("ALA", "GLY"), self_template=(ComponentRef("A", 1),)).features
        assert input.template_mask[0, 0].tolist() == [0, 1]


class TestConditions:
    def test_contact_classes(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        contact = Boltz2Contact(ComponentRef("A", 0), AtomSite("B", 0, "C"), force=True)
        prepared = Boltz2Processor().prepare_input(structure, contacts=(contact,))
        token = int(prepared.context.slots.layout.atom_token[structure.system.atom_index([contact.endpoint_b])[0]])
        classes = prepared.features.contact_conditioning[0].argmax(-1)
        assert classes[0, token] == 4 and classes[1, token] == 1
        assert prepared.features.contact_pair_index.shape[-1] == int((prepared.context.slots.token == 0).sum())

    def test_residue_atom_refused(self, peptide):
        contact = Boltz2Contact(AtomSite("A", 0, "CA"), ComponentRef("A", 1))
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            Boltz2Processor().prepare_input(peptide("ALA", "GLY"), contacts=(contact,))


class TestAffinity:
    def test_ligand_tokens(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        prepared = Boltz2AffinityProcessor().prepare_input(structure)
        components = prepared.context.slots.layout.token_component
        assert prepared.features.affinity_token_mask[0].tolist() == (components == 2).tolist()
        assert prepared.features.affinity_mw is not None

    def test_crop_around_ligand(self, peptide):
        layout = Boltz2AffinityProcessor().prepare_input(peptide(*["GLY"] * 210, ligands=("ACT",))).context.slots.layout
        assert 209 in layout.token_component and 0 not in layout.token_component  # the residues nearest the ligand

    def test_query_row_profile(self, peptide):
        msa = {"A": MSA("P", ("AG", "GA"), np.zeros((2, 2), dtype=np.uint32), ("AG",), np.zeros((1, 2), np.uint32))}
        input = Boltz2AffinityProcessor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)), msa=msa).features
        assert input.msa_profile[0, 0, TOKEN_INDEX["A"]] == 0.5
        assert input.affinity_msa_profile[0, 0, TOKEN_INDEX["A"]] == 1

    def test_ambiguous_ligand_refused(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            Boltz2AffinityProcessor().prepare_input(peptide("ALA", ligands=("ACT", "ACT")))

    def test_canonical_rank_names(self):
        from rdkit import Chem

        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        ranks = Chem.CanonicalRankAtoms(molecule)
        expected = sorted(f"{atom.GetSymbol().upper()}{rank + 1}" for atom, rank in zip(molecule.GetAtoms(), ranks))
        assert sorted(standardize_boltz2_affinity_smiles("OCC.[Na+]").atom_names) == expected

    def test_upstream_ligand_id(self):  # upstream's LIG1; LIG is a CCD id
        assert standardize_boltz2_affinity_smiles("OCC").component_id == "LIG1"
