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

"""Behavior of ofoldx.models.protenix.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError, StructureRef
from ofoldx.models.protenix.config import ProtenixProcessorConfig
from ofoldx.models.protenix.model import ProtenixAtomContact, ProtenixTokenContact
from ofoldx.models.protenix.processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
)
from ofoldx.msa import MSA, MsaCollection

_PROFILE = 384 + len(TOKENS)


@pytest.fixture
def processor():
    return ProtenixProcessor(ProtenixProcessorConfig(augment_reference_positions=False), dtype=torch.float64)


def pool(*rows: str) -> MSA:
    r"""A pool of entity ``P``: the query, then each unpaired row."""
    counts = np.zeros((len(rows), len(rows[0])), dtype=np.uint32)
    return MSA("P", rows, counts, rows[:1], counts[:1])


def plan(*rows: str) -> MsaCollection:
    r"""An exact plan over one pool of entity ``P``: its paired query, then each unpaired row."""
    paired = np.zeros((len(rows), 1), dtype=bool)
    paired[0] = True
    return MsaCollection((pool(*rows),), np.arange(len(rows))[:, None], paired)


class TestPrepare:
    def test_prediction_scope(self, processor, peptide):
        prepared = processor.prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert prepared.features.token_features.shape[-1] == 384 + 2 * len(TOKENS) + 1

    def test_designated_leaving_addressed(self, processor, peptide):
        structure = peptide("ALA", "GLY")
        prepared = processor.prepare_input(structure)
        names = structure.system.text("atom", structure.system.atom_symbol[prepared.context.slots.atom]).tolist()
        assert "OXT" in names

    def test_token_bonds(self, processor, peptide):
        bonds = processor.prepare_input(peptide("ALA", "GLY", ligands=("ACT",))).features.token_bonds[0]
        assert bonds[:2, :2].sum() == 0  # the junction of chain neighbours, as upstream
        assert bonds[2:, 2:].any()  # the ligand's own bonds

    def test_crop_needs_generator(self, processor, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            processor.prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, processor, peptide):
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure, generator=np.random.default_rng(3))
        tensors = target
        assert tensors.coordinates.shape == (1, len(structure.coordinates), 3)
        assert tensors.resolution_mask.tolist() == [False]


class TestMsa:
    def test_profile_unique_rows(self, processor, peptide):
        prepared = processor.prepare_input(peptide(*["ALA"] * 5), msa=plan("AAAAA", "GAAAA", "GAAAA"))
        assert prepared.features.msa_tokens.shape[-2] == 3
        profile = prepared.features.token_features[0, 0, _PROFILE : _PROFILE + len(TOKENS)]
        assert profile[TOKEN_INDEX["A"]] == profile[TOKEN_INDEX["G"]] == 0.5

    def test_short_chain_query_only(self, processor, peptide):
        prepared = processor.prepare_input(peptide("ALA", "ALA"), msa=plan("AA", "GA"))
        assert prepared.features.msa_tokens.shape[-2] == 2
        assert (prepared.features.msa_tokens[0, 1:] == TOKEN_INDEX["-"]).all()
        assert prepared.features.token_features[0, 0, _PROFILE + TOKEN_INDEX["A"]] == 1

    def test_short_chain_pool_ignored(self, processor, peptide):
        prepared = processor.prepare_input(peptide("ALA", "ALA"), msa={"A": pool("AA", "GA", "CA")})
        assert prepared.features.msa_tokens.tolist() == [[[TOKEN_INDEX["A"]] * 2]]

    def test_letters_read_upstream(self, processor, peptide):
        prepared = processor.prepare_input(peptide(*["ALA"] * 5), msa=plan("AAAAA", "BZUJO"))
        assert prepared.features.msa_tokens[0, 1].tolist() == [TOKEN_INDEX[letter] for letter in "DECXX"]


class TestConstraints:
    def test_atom_contact(self, peptide):
        structure = peptide("ALA", "GLY")
        contact = ProtenixAtomContact(AtomSite("A", 0, "N"), AtomSite("A", 0, "CA"), 4.0)
        prepared = ProtenixConstraintProcessor(dtype=torch.float64).prepare_input(structure, atom_contacts=(contact,))
        layout = prepared.context.slots.layout
        first, second = layout.atom_token[structure.system.atom_index([contact.atom_a, contact.atom_b])]
        assert first != second  # the named residue is tokenized per atom
        assert prepared.features.atom_contact[0, first, second].tolist() == [4.0, 0.0]
        assert prepared.features.token_bonds[0, first, second] == 0  # inside a standard residue, as upstream

    @pytest.mark.parametrize("kind", ["atom_contacts", "token_contacts"])
    def test_contact_to_excluded_hydrogen_names_requested_atom(self, peptide, kind):
        structure = peptide("ALA", "ALA")
        hydrogen = AtomSite("A", 0, "HB3")
        atom = int(structure.system.atom_index([hydrogen])[0])
        assert atom >= len(structure.system.component_symbol)
        contact = (
            ProtenixAtomContact(hydrogen, AtomSite("A", 1, "CA"), 4.0)
            if kind == "atom_contacts"
            else ProtenixTokenContact(hydrogen, ComponentRef("A", 1), 4.0)
        )
        with pytest.raises(InvalidRequestError) as error:
            ProtenixConstraintProcessor().prepare_input(structure, **{kind: (contact,)})
        assert error.value.code == "request_invalid_positions"
        assert error.value.refs == (StructureRef(atom=atom),)
        assert error.value.detail["argument"] == kind
        assert error.value.detail["reason"] == "atom_not_tokenized"


class TestTfg:
    def test_ligand_geometry(self, processor, peptide):
        prepared = processor.prepare_input(peptide("ALA", ligands=("ACT",)), tfg=True)
        features = prepared.features.tfg_features
        ligand = prepared.context.slots.layout.token_component[prepared.context.slots.token] == 1
        assert ligand[features.pairwise_distance_index.cpu().numpy()].all()
        assert features.planar_improper_index.shape[-1] == 3  # the carboxylate carbon


@pytest.mark.parametrize(
    "processor_type",
    [ProtenixProcessor, ProtenixStructureConditionProcessor, ProtenixConstraintProcessor, ProtenixTemplateProcessor],
)
def test_variant_composition_and_independent_truth(peptide, processor_type):
    structure = peptide("ALA", "GLY")
    arrays = structure.build()
    arrays.columns["coordinates"] += [15, -4, 8]
    target = Structure.from_arrays(structure.system, arrays)
    processor = processor_type(dtype=torch.float64)
    prepared = processor.prepare_input(structure, generator=np.random.default_rng(3))
    labels = processor.prepare_target(target, prepared=prepared)
    combined, combined_labels = processor.prepare(structure, target=target, generator=np.random.default_rng(3))
    torch.testing.assert_close(prepared.features.to_dict(), combined.features.to_dict())
    torch.testing.assert_close(labels.to_dict(), combined_labels.to_dict())
    mask = labels.coordinate_mask[0]
    torch.testing.assert_close(labels.coordinates[0, mask], torch.from_numpy(target.coordinates.copy())[mask])
