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

"""Behavior of ofoldx.models.alphafold2.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.alphafold2 import AlphaFold2Casp14Pipeline


class TestFold:
    @pytest.mark.parametrize("kind", ["monomer", "multimer"])
    def test_reproducible_sampling(self, kind, tiny_alphafold2, chains):
        pipeline = tiny_alphafold2(kind)
        structure = chains(A=("ALA", "GLY"), B=("GLY",)) if kind == "multimer" else chains(A=("ALA", "GLY"))
        first = pipeline(structure, num_samples=2, seed=720)
        second = pipeline(structure, num_samples=2, seed=720)
        assert len(first.predictions) == 2
        for expected, actual in zip(first.predictions, second.predictions, strict=True):
            np.testing.assert_array_equal(actual.structure.coordinates, expected.structure.coordinates)
        axis = first.predictions[0].metrics["plddt"].axis
        assert axis[:2] == (ComponentRef("A", 0), ComponentRef("A", 1))

    def test_decoded_onto_target(self, tiny_alphafold2, peptide):
        pipeline = tiny_alphafold2("monomer")
        prepared = pipeline.prepare(peptide("ALA", "GLY"))
        raw = pipeline.forward(prepared.features, generator=torch.Generator().manual_seed(94))
        decoded = pipeline.postprocess(prepared, raw).predictions[0].structure
        atom = prepared.context.slots.atom
        valid = (raw[0].final_atom_mask[0].reshape(-1).numpy() > 0) & (atom >= 0)
        positions = raw[0].final_atom_positions[0].reshape(-1, 3).double().numpy()
        np.testing.assert_array_equal(decoded.coordinates[atom[valid]], positions[valid])
        assert (decoded.coordinate_evidence[atom[valid]] == CoordinateEvidence.PREDICTED).all()
        assert decoded.coordinate_mask.sum() == valid.sum()  # hydrogens and OXT are not predicted


class TestInitialGuess:
    def test_binder_scores(self, tiny_alphafold2, chains):
        pipeline = tiny_alphafold2("initial_guess")
        result = pipeline(chains(A=("ALA", "GLY"), B=("GLY", "ALA")), binder_instance_id="B", seed=57)
        scores = result.predictions[0].scores
        assert scores["binder_mean_plddt"].scope == frozenset({ComponentRef("B", 0), ComponentRef("B", 1)})
        assert {"ptm", "iptm", "interface_pae", "normalized_interface_pae"} <= scores.keys()


class TestCasp14:
    def test_ranked_by_plddt(self, tiny_alphafold2, chains):
        child = tiny_alphafold2("monomer", num_ensembles=8)
        pipeline = AlphaFold2Casp14Pipeline(model_1=child, model_2=child, model_3=child, model_4=child, model_5=child)
        result = pipeline(chains(A=("ALA", "GLY")), seed=720, models_to_relax="none")
        scores = [item.prediction.scores["mean_plddt"].value for item in result.ranked]
        assert scores == sorted(scores, reverse=True)
        assert {item.model_name for item in result.ranked} == {f"model_{k}" for k in range(1, 6)}
        assert all(item.relaxation is None for item in result.ranked)
