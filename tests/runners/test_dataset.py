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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ofoldx.data.structure import Structure
from ofoldx.errors import InvalidRequestError
from ofoldx.io import save_structure
from ofoldx.models.alphafold3 import AlphaFold3Processor
from ofoldx.runners.dataset import StructureDataset


def _identity(structure: Structure) -> Structure:
    return structure


def _pools(structure: Structure, *, msa: Any) -> Any:
    return msa


@pytest.fixture
def artifact(tmp_path, peptide) -> Path:
    """A native artifact of chain A (ALA, GLY) and a ligand B."""
    path = tmp_path / "sample.safetensors"
    save_structure(peptide("ALA", "GLY", ligands=("ACT",)), path)
    return path


@pytest.fixture
def manifest(tmp_path):
    def write(*records: dict[str, Any]) -> Path:
        path = tmp_path / "records.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    return write


class TestManifest:
    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"msas": {}}, "unknown dataset record key"),
            ({"msa": [{}]}, "must be a JSON object"),
            ({"target": "target.safetensors"}, "requires `path`"),
            ({"target": {"path": "target.safetensors", "instance_ids": [True]}}, "must be a list of strings"),
        ],
    )
    def test_schema(self, artifact, manifest, record, message) -> None:
        with pytest.raises(ValueError, match=message):
            StructureDataset(manifest({"structure": str(artifact), **record}), _identity)

    def test_native_structure(self, artifact, manifest, peptide) -> None:
        value, index, path = StructureDataset(manifest({"structure": artifact.name}), _identity)[0]

        assert value.digest() == peptide("ALA", "GLY", ligands=("ACT",)).digest()
        assert (index, path) == (0, str(artifact))

    def test_target(self, artifact, manifest, peptide) -> None:
        source = peptide("ALA", "GLY", ligands=("ACT",))
        arrays = source.build()
        arrays.columns["coordinates"] += np.array([11.0, 17.0, 23.0])
        truth = Structure.from_arrays(source.system, arrays)
        target_path = artifact.with_name("truth.safetensors")
        save_structure(truth, target_path)
        record = {"structure": artifact.name, "target": {"path": target_path.name}}
        processor = AlphaFold3Processor()
        (prepared, target), _, _ = StructureDataset(manifest(record), processor.prepare)[0]

        mask = target.coordinate_mask[0].cpu().numpy()
        assert mask.any()
        np.testing.assert_array_equal(target.coordinates[0, mask].cpu().numpy(), truth.coordinates[mask])
        np.testing.assert_array_equal(prepared.context.structure.coordinates, source.coordinates)

    def test_unlabelled_sample(self, artifact):
        (prepared, target), _, _ = StructureDataset(artifact, AlphaFold3Processor().prepare)[0]
        assert target is None
        assert prepared.context.structure.coordinates.shape[-1] == 3

    def test_msa_record(self, tmp_path, artifact, manifest) -> None:
        (tmp_path / "chain.a3m").write_text(">query\nAG\n>hit\nAA\n")
        record = {"structure": artifact.name, "msa": {"path": "chain.a3m", "entity_id": "P"}}

        pools, _, _ = StructureDataset(manifest(record), _pools)[0]

        assert list(pools) == ["A"]
        assert pools["A"].unpaired_sequences == ("AG", "AA")


class TestStructureDataset:
    def test_indexed_msa(self, tmp_path, artifact) -> None:
        directory = tmp_path / "msa" / "1"
        directory.mkdir(parents=True)
        (directory / "pairing.a3m").write_text(">query\nAG\n>UniRef100_P1_9606/\nGG\n")
        (directory / "non_pairing.a3m").write_text(">query\nAG\n")

        pools, _, _ = StructureDataset(artifact, _pools, msa_index={"AG": "1"}, msa_root=tmp_path / "msa")[0]

        assert pools["A"].unpaired_sequences == ("AG", "GG")

    def test_directory_source(self, tmp_path: Path) -> None:
        for name in ("first.cif.gz", "second.pdb", "third.safetensors", "ignored.txt"):
            (tmp_path / name).touch()

        dataset = StructureDataset(tmp_path, _identity)

        assert dataset.records == [str(tmp_path / "third.safetensors")]

    @pytest.mark.parametrize("source", ["file", "manifest", "directory"])
    def test_refuses_parsed_structure(self, tmp_path, manifest, source) -> None:
        structure = tmp_path / "sample.cif"
        structure.touch()
        named = tmp_path if source == "directory" else structure

        with pytest.raises(InvalidRequestError, match="dataset_not_native") as error:
            StructureDataset(manifest({"structure": structure.name}) if source == "manifest" else named, _identity)

        assert str(named) in str(error.value)

    def test_error_context(self, artifact) -> None:
        def fail(_structure: Structure) -> None:
            raise ValueError("processor failed")

        with pytest.raises(ValueError, match="processor failed") as error:
            StructureDataset(artifact, fail)[0]

        assert error.value.__notes__ == [f"dataset index=0, structure={artifact}"]

    def test_epoch(self, artifact) -> None:
        def random(_structure: Structure, *, generator: np.random.Generator) -> float:
            return float(generator.random())

        dataset = StructureDataset(artifact, random, seed=17)
        first = dataset[0][0]
        assert first == dataset[0][0]

        dataset.set_epoch(1)
        assert first != dataset[0][0]

        restored = StructureDataset(artifact, random, seed=17)
        restored.load_state_dict(dataset.state_dict())
        assert restored[0][0] == dataset[0][0]
