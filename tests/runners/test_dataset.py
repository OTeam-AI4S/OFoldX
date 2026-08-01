from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from ofoldx.data import MsaCollection, Structure, save_structure
from ofoldx.runners.dataset import StructureDataset

STRUCTURE = Path(__file__).resolve().parents[2] / "examples" / "data" / "tripeptide.pdb"


def _identity(structure: Structure) -> Structure:
    return structure


class TestStructureDataset:
    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"structure": str(STRUCTURE), "msas": {}}, "unknown dataset record key"),
            ({"structure": str(STRUCTURE), "msa": [{}]}, "must be a JSON object"),
            ({"structure": str(STRUCTURE), "target": str(STRUCTURE)}, "must contain exactly"),
            (
                {"structure": str(STRUCTURE), "target": {"path": str(STRUCTURE), "instance_ids": [True]}},
                "must be a list of strings",
            ),
        ],
    )
    def test_manifest_schema(self, tmp_path: Path, record: dict[str, Any], message: str) -> None:
        manifest = tmp_path / "records.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            StructureDataset(manifest, _identity)

    def test_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "records.jsonl"
        manifest.write_text(json.dumps({"structure": str(STRUCTURE)}) + "\n", encoding="utf-8")

        value, index, path = StructureDataset(manifest, _identity)[0]

        assert isinstance(value, Structure)
        assert index == 0
        assert path == str(STRUCTURE)

    def test_target(self, tmp_path: Path) -> None:
        structure_path = tmp_path / "structure.structure.safetensors"
        target_path = tmp_path / "target.structure.safetensors"
        save_structure(Structure.from_sequence("A"), structure_path)
        save_structure(Structure.from_sequence("GG"), target_path)
        manifest = tmp_path / "records.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "structure": structure_path.name,
                    "target": {"path": target_path.name, "instance_ids": ["A"]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def processor(
            structure: Structure,
            *,
            target: Structure,
            generated_target_instance_ids: list[str],
        ) -> tuple[Structure, Structure, list[str]]:
            return structure, target, generated_target_instance_ids

        (structure, target, rows), _, _ = StructureDataset(manifest, processor)[0]

        assert structure.system.component_symbol.numel() == 1
        assert target.system.component_symbol.numel() == 2
        assert rows == ["A"]

    def test_msa(self, tmp_path: Path) -> None:
        msa_root = tmp_path / "msa"
        msa_directory = msa_root / "1"
        msa_directory.mkdir(parents=True)
        (msa_directory / "uniref100_hits.a3m").write_text(">query\nAGS\n>UniRef100_hit_MOUSE\nGGS\n")
        (msa_directory / "mmseqs_other_hits.a3m").write_text(">query\nAGS\n")

        def processor(value: Structure, *, msa: MsaCollection) -> tuple[Structure, MsaCollection]:
            return value, msa

        value, _, _ = StructureDataset(
            STRUCTURE,
            processor,
            msa_index={"AGS": "1"},
            msa_root=msa_root,
        )[0]

        structure, msa = value
        assert isinstance(structure, Structure)
        assert msa.msas[0].unpaired_sequences == ("AGS", "GGS")

    def test_prepared_structure(self, tmp_path: Path) -> None:
        from ofoldx.io import read_structure

        structure = read_structure(STRUCTURE)
        path = tmp_path / "sample.structure.safetensors"
        save_structure(structure, path)

        value, _, source = StructureDataset(path, _identity)[0]

        torch.testing.assert_close(value.coordinates, structure.coordinates)
        assert source == str(path)

    def test_compressed_mmcif(self, tmp_path: Path) -> None:
        expected = {tmp_path / "first.cif.gz", tmp_path / "second.mmcif.gz"}
        for path in expected:
            path.touch()
        (tmp_path / "ignored.txt").touch()

        dataset = StructureDataset(tmp_path, _identity)

        assert {Path(record) for record in cast(list[str], dataset.records)} == expected

    def test_error_context(self) -> None:
        def fail(_structure: Structure) -> None:
            raise ValueError("processor failed")

        with pytest.raises(ValueError, match="processor failed") as error:
            StructureDataset(STRUCTURE, fail)[0]

        assert error.value.__notes__ == [f"dataset index=0, structure={STRUCTURE}"]

    def test_epoch(self) -> None:
        def random(_structure: Structure, *, generator: torch.Generator) -> torch.Tensor:
            return torch.rand((), generator=generator)

        dataset = StructureDataset(STRUCTURE, random, seed=17)
        first = dataset[0][0]
        assert torch.equal(first, dataset[0][0])

        dataset.set_epoch(1)
        assert not torch.equal(first, dataset[0][0])

        restored = StructureDataset(STRUCTURE, random, seed=17)
        restored.load_state_dict(dataset.state_dict())
        assert torch.equal(restored[0][0], dataset[0][0])
