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

"""Behavior of ofoldx.models.alphafold3.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.structure import Structure, StructureArrays
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InsufficientInformationError, InvalidRequestError
from ofoldx.models.alphafold3.processor import ALPHAFOLD3_TEMPLATE_SLOTS, AlphaFold3Processor
from ofoldx.msa import MSA, MsaCollection
from ofoldx.provenance import Basis, Resolution, state
from ofoldx.templates import Template, TemplateAlignment, chain_coordinates

_GAP = TOKEN_INDEX["-"]


@pytest.fixture
def pool():
    """``pool(*rows)``: the MSA pool of entity ``P`` whose unpaired rows are ``rows`` (the first is the query)."""

    def build(*rows: str) -> MSA:
        counts = np.zeros((len(rows), len(rows[0])), dtype=np.uint32)
        return MSA("P", rows, counts, rows[:1], counts[:1])

    return build


@pytest.fixture
def template(peptide):
    """``template(*components)``: ``peptide(*components)`` as author chain ``T``, aligned position by position onto
    entity ``P``."""

    def build(*components: str) -> Template:
        structure = peptide(*components)
        arrays = structure.build()
        arrays.columns["instance_author_chain"][:] = 0
        arrays.columns["instance_author_chain_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        source = Structure.from_arrays(
            structure.system, StructureArrays({**arrays.symbols, "chain": ("T",)}, arrays.columns)
        )
        pairs = tuple((k, k) for k in range(len(components)))
        return Template(chain_coordinates(source, ["T"]), (TemplateAlignment("P", None, "T", pairs),))

    return build


class TestPrepare:
    def test_prediction_scope(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = AlphaFold3Processor().prepare_input(structure)
        system = structure.system
        assert prepared.context.scope.rule == "all" and prepared.context.scope.frame == "new"
        assert "OXT" in system.text("atom", system.atom_symbol[prepared.context.slots.atom]).tolist()
        assert not (system.atomic_number[prepared.context.slots.atom] == 1).any()

    def test_ion_has_no_frame(self, peptide):
        prepared = AlphaFold3Processor().prepare_input(peptide("ALA", "GLY", ligands=("ZN",)))
        assert prepared.features.frame_mask[0].tolist() == [True, True, False]

    def test_ligand_reads_gaps(self, peptide):
        rows = AlphaFold3Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",))).features.msa_tokens[0]
        assert rows[0, :2].tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["G"]]
        assert (rows[:, 2:] == _GAP).all()

    def test_intra_ligand_no_bonds(self, peptide):
        prepared = AlphaFold3Processor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        assert not prepared.features.token_bonds.any()

    def test_profile_over_unpaired(self, peptide, pool):
        prepared = AlphaFold3Processor().prepare_input(peptide(*["ALA"] * 5), msa={"A": pool("AAAAA", "GAAAA")})
        profile = prepared.features.token_features[0, :, 384 + len(TOKENS) : 384 + 2 * len(TOKENS)]
        assert profile[0, TOKEN_INDEX["A"]] == profile[0, TOKEN_INDEX["G"]] == 0.5
        assert prepared.features.msa_tokens[0, 1, 0] == TOKEN_INDEX["G"]

    @pytest.mark.parametrize("explicit_plan", [False, True], ids=["mapping", "plan"])
    def test_short_chain_query_only(self, peptide, pool, explicit_plan):
        alignment = pool("AG", "GG")
        msa = (
            MsaCollection((alignment,), np.array([[0], [1]]), np.array([[True], [False]]))
            if explicit_plan
            else {"A": alignment}
        )
        rows = AlphaFold3Processor().prepare_input(peptide("ALA", "GLY"), msa=msa).features.msa_tokens[0]
        assert rows[0].tolist() == [TOKEN_INDEX["A"], TOKEN_INDEX["G"]]
        assert len(rows) == (2 if explicit_plan else 1)
        assert (rows[1:] == _GAP).all()

    @pytest.mark.parametrize("length, filled", [(5, True), (4, False)])
    def test_template_slots(self, peptide, template, length, filled):
        components = ("ALA", "GLY") * 2 + ("ALA",) * (length - 4)
        features = (
            AlphaFold3Processor().prepare_input(peptide(*components), templates=(template(*components),)).features
        )
        features = features.template_features
        residues = features.residue_type_ids[0]
        assert residues.shape == (ALPHAFOLD3_TEMPLATE_SLOTS, length)
        assert (residues[1:] == TOKEN_INDEX["A"]).all() and not features.backbone_frame_mask[0, 1:].any()
        if filled:
            assert residues[0].tolist() == [TOKEN_INDEX["A" if c == "ALA" else "G"] for c in components]
            assert features.backbone_frame_mask[0, 0].all()
        else:
            assert (residues[0] == TOKEN_INDEX["A"]).all() and not features.backbone_frame_mask[0, 0].any()

    def test_unknown_polymer_type_refused(self, peptide):
        structure = peptide("ALA", "GLY")
        arrays = structure.system.build()
        arrays.columns["entity_polymer_type"][0] = -1
        arrays.columns["entity_polymer_type_state"][0] = state(Resolution.UNKNOWN, Basis.NONE)
        unknown = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())
        with pytest.raises(InsufficientInformationError, match="prepare_unknown_feature"):
            AlphaFold3Processor().prepare_input(unknown)

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            AlphaFold3Processor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = AlphaFold3Processor(dtype=torch.float64)
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure)
        tensors = target
        assert tensors.coordinates.shape == (1, len(structure.coordinates), 3)
        assert tensors.coordinate_mask[0].nonzero().flatten().tolist() == sorted(prepared.context.slots.atom.tolist())


class TestPreparationInterface:
    def test_composition_and_existing_correspondence(self, peptide):
        structure = peptide("ALA", "GLY")
        processor = AlphaFold3Processor()
        calls = []

        def first(structure, layout, *, required, generator):
            calls.append(True)
            keep = np.zeros(len(layout.token_component), dtype=bool)
            keep[0] = True
            return keep

        prepared = processor.prepare_input(structure, cropper=first, generator=np.random.default_rng(7))
        snapshot = prepared.features.to_dict()
        target = processor.prepare_target(structure, prepared=prepared)
        assert len(calls) == 1
        torch.testing.assert_close(prepared.features.to_dict(), snapshot)
        combined, combined_target = processor.prepare(
            structure, target=structure, cropper=first, generator=np.random.default_rng(7)
        )
        assert len(calls) == 2
        torch.testing.assert_close(prepared.features.to_dict(), combined.features.to_dict())
        torch.testing.assert_close(target.to_dict(), combined_target.to_dict())
        input_only, no_target = processor.prepare(structure)
        assert input_only.context.structure is structure
        assert no_target is None

    def test_truth_precision_is_independent_of_activation_dtype(self, peptide):
        structure = peptide("ALA", "GLY")
        processor = AlphaFold3Processor(dtype=torch.float16)
        prepared = processor.prepare_input(structure)
        arrays = structure.build()
        arrays.columns["coordinates"][structure.coordinate_mask] += 100_000
        truth = Structure.from_arrays(structure.system, arrays)
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.coordinates.dtype == torch.float64
        assert torch.isfinite(target.coordinates).all()
        mask = target.coordinate_mask[0]
        torch.testing.assert_close(target.coordinates[0, mask], torch.from_numpy(truth.coordinates.copy())[mask])

    def test_unaddressed_large_truth_is_not_lowered(self, peptide):
        structure = peptide("ALA", "GLY")
        processor = AlphaFold3Processor(dtype=torch.float16)
        prepared = processor.prepare_input(structure)
        excluded = prepared.context.slots.unaddressed
        assert len(excluded)
        arrays = structure.build()
        arrays.columns["coordinates"][excluded] = 1e300
        truth = Structure.from_arrays(structure.system, arrays)
        target = processor.prepare_target(truth, prepared=prepared)
        assert not target.coordinate_mask[0, excluded].any()
        assert not target.coordinates[0, excluded].any()
        assert torch.isfinite(target.coordinates).all()
