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

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MonomerType
from ofoldx.errors import ContradictionError, InsufficientInformationError
from ofoldx.msa import MSA, MsaCollection, save_msa_collection
from ofoldx.pipelines.input import entity_pools, resolve_msa_input, resolve_polymer_input


@pytest.fixture
def system(polymers):
    """Proteins A and B (entity P, AG) and a ligand L."""
    return polymers(("A", "P", ("ALA", "GLY")), ("B", "P", ("ALA", "GLY")), ("L", "Z", ("ZN",)))


@pytest.fixture
def a3m(tmp_path):
    path = tmp_path / "input.a3m"
    path.write_text(">query\nAG\n>first\nAA\n>second\nGG\n")
    return path


class TestPolymerInput:
    def test_inline_protein(self, library):
        structure = resolve_polymer_input("GAK", library=library)

        system = structure.system
        assert system.text("instance", system.instance_symbol).tolist() == ["chain_1"]
        assert system.text("component", system.component_symbol).tolist() == ["GLY", "ALA", "LYS"]

    def test_fasta_entities(self, tmp_path, library):
        path = tmp_path / "input.faa"
        path.write_text(">first\nGA\n>second\nKA\n>third\nGA\n")

        system = resolve_polymer_input(path, library=library).system

        assert system.text("instance", system.instance_symbol).tolist() == ["first", "second", "third"]
        assert system.instance_entity.tolist() == [0, 1, 0]

    @pytest.mark.parametrize("text", [">A\nGA\n>A\nKA\n", ">chain_2\nGA\n>\nKA\n"])
    def test_duplicate_name(self, tmp_path, library, text):
        path = tmp_path / "input.faa"
        path.write_text(text)

        with pytest.raises(ContradictionError) as caught:
            resolve_polymer_input(path, library=library)
        assert caught.value.code == "sequence_name_duplicate"

    def test_nucleotide_fasta(self, tmp_path, invalid):
        path = tmp_path / "input.fna"
        path.write_text(">chain\nACG\n")

        invalid(resolve_polymer_input, path, code="format_unknown")

    def test_structure(self, peptide):
        structure = peptide("ALA")

        assert resolve_polymer_input(structure) is structure


class TestMsaInput:
    def test_every_instance(self, a3m, system):
        pools = resolve_msa_input(a3m, system, monomer_types={MonomerType.PROTEIN})

        assert sorted(pools) == ["A", "B"]
        assert {pool.entity_id for pool in pools.values()} == {"P"}
        assert {pool.unpaired_sequences for pool in pools.values()} == {("AG", "AA", "GG")}

    def test_ambiguous_entity(self, a3m, polymers):
        system = polymers(("A", "P", ("ALA", "GLY")), ("B", "Q", ("GLY", "ALA")))

        with pytest.raises(InsufficientInformationError) as caught:
            resolve_msa_input(a3m, system, monomer_types={MonomerType.PROTEIN})
        assert caught.value.code == "msa_entity_unresolved"
        assert len(caught.value.candidates) == 2

    def test_native_collection(self, tmp_path, system):
        counts = np.zeros((2, 2), dtype=np.uint32)
        msa = MSA("P", ("AG", "AA"), counts, ("AG",), counts[:1])
        path = tmp_path / "input.msa.safetensors"
        save_msa_collection(MsaCollection((msa,), np.array([[0], [1]]), np.array([[True], [False]])), path)

        restored = resolve_msa_input(path, system, monomer_types={MonomerType.PROTEIN})

        assert isinstance(restored, MsaCollection)
        assert restored.msas[0].unpaired_sequences == ("AG", "AA")

    def test_unknown_entity(self, system, invalid):
        counts = np.zeros((1, 2), dtype=np.uint32)

        invalid(entity_pools, MSA("X", ("AG",), counts, ("AG",), counts), system, code="msa_entity_unknown")
