from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofoldx.processor import UnresolvedTrainingCropError
from ofoldx.runners.dataset import StructureDataset
from ofoldx.runners.runner import Runner


def _load_train_script() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("ofoldx_test_train_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_requires_index_and_msa_path_together(capsys: pytest.CaptureFixture[str]) -> None:
    train = _load_train_script()
    with pytest.raises(SystemExit, match="2"):
        train.parse_args(["train.cif", "--seq-to-pdb-index", "index.json"])
    assert "must be provided together" in capsys.readouterr().err


def test_train_cli_routes_indexed_msa_to_train_and_all_evaluation_splits() -> None:
    train = _load_train_script()
    args = train.parse_args(
        [
            "train.cif",
            "--val-source",
            "val.cif",
            "--eval-source",
            "benchmark=test.cif",
            "--seq-to-pdb-index",
            "index.json",
            "--msa-path",
            "msa",
        ]
    )
    config = train.build_config(args)
    assert config["data"]["train"] == {
        "source": "train.cif",
        "seq_to_pdb_index": "index.json",
        "msa_path": "msa",
    }
    assert config["data"]["val"] == {
        "source": "val.cif",
        "seq_to_pdb_index": "index.json",
        "msa_path": "msa",
    }
    assert config["data"]["benchmark"] == {
        "source": "test.cif",
        "seq_to_pdb_index": "index.json",
        "msa_path": "msa",
    }


def test_train_cli_routes_cluster_indices_file_and_requires_cluster_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train = _load_train_script()
    with pytest.raises(SystemExit, match="2"):
        train.parse_args(["train.cif", "--cluster-indices-file", "indices.jsonl"])
    assert "require --cluster-file" in capsys.readouterr().err
    args = train.parse_args(
        ["train.cif", "--cluster-file", "clusters.txt", "--cluster-indices-file", "indices.jsonl"]
    )
    assert train.build_config(args)["data"]["train"]["cluster_sampling"]["indices_file"] == "indices.jsonl"


def test_structure_dataset_loads_indexed_msa_and_passes_it_to_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure_path = tmp_path / "sample.cif"
    structure_path.touch()
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"SEQUENCE": "directory"}), encoding="utf-8")
    msa_path = tmp_path / "msa"
    msa_path.mkdir()
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[1]))
    msa = object()
    calls: list[tuple[Any, ...]] = []

    def read_structure(path: str, **kwargs: Any) -> object:
        calls.append(("structure", path, kwargs))
        return structure

    def read_msa(*args: Any, **kwargs: Any) -> object:
        calls.append(("msa", *args, kwargs))
        return msa

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    monkeypatch.setattr("ofoldx.io.read_msa", read_msa)
    processor_calls: list[tuple[Any, dict[str, Any]]] = []
    dataset = StructureDataset(
        structure_path,
        lambda value, **kwargs: processor_calls.append((value, kwargs)) or "processed",
        seq_to_pdb_index=index_path,
        msa_path=msa_path,
        processor_kwargs={"max_tokens": 128},
    )

    assert dataset[0] == "processed"
    assert calls[0] == (
        "structure",
        str(structure_path),
        {"ccd_revision": None, "ccd_snapshot_path": None},
    )
    assert calls[1][0:2] == ("msa", structure)
    assert calls[1][2] == {"SEQUENCE": "directory"}
    assert calls[1][3] == msa_path.resolve()
    assert processor_calls == [(structure, {"msa": msa, "max_tokens": 128})]


def test_structure_dataset_skips_indexed_msa_for_structure_without_protein(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure_path = tmp_path / "ligand.cif"
    structure_path.touch()
    index_path = tmp_path / "index.json"
    index_path.write_text("{}", encoding="utf-8")
    msa_path = tmp_path / "msa"
    msa_path.mkdir()
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[4]))

    monkeypatch.setattr("ofoldx.io.read_structure", lambda path, **kwargs: structure)

    def reject_read_msa(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("indexed MSA loading must be skipped for non-protein structures")

    monkeypatch.setattr("ofoldx.io.read_msa", reject_read_msa)
    dataset = StructureDataset(
        structure_path,
        lambda value, **kwargs: (value, kwargs),
        seq_to_pdb_index=index_path,
        msa_path=msa_path,
    )

    assert dataset[0] == (structure, {})


def test_manifest_msa_takes_precedence_over_indexed_msa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    structure_path = tmp_path / "sample.cif"
    structure_path.touch()
    explicit_msa = tmp_path / "sample.a3m"
    explicit_msa.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"structure": structure_path.name, "msa": {"path": explicit_msa.name, "entity_id": "1"}}),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    index_path.write_text("{}", encoding="utf-8")
    msa_path = tmp_path / "msa"
    msa_path.mkdir()
    structure = object()
    explicit = object()
    events: list[str] = []

    monkeypatch.setattr(
        "ofoldx.io.read_structure",
        lambda path, **kwargs: events.append("structure") or structure,
    )

    def read_msa(path: Any, *args: Any, **kwargs: Any) -> object:
        events.append("explicit")
        assert path == str(explicit_msa)
        assert not args
        assert kwargs == {"entity_id": "1"}
        return explicit

    monkeypatch.setattr("ofoldx.io.read_msa", read_msa)
    dataset = StructureDataset(
        manifest,
        lambda value, **kwargs: (value, kwargs),
        seq_to_pdb_index=index_path,
        msa_path=msa_path,
    )

    assert dataset[0] == (structure, {"msa": explicit})
    assert events == ["structure", "explicit"]


def test_structure_dataset_passes_the_ccd_locator_to_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure_path = tmp_path / "sample.cif"
    structure_path.touch()
    captured: dict[str, Any] = {}

    def read_structure(path: str, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    dataset = StructureDataset(
        structure_path,
        lambda structure: structure,
        ccd_revision="ccd-revision",
    )

    dataset[0]
    assert captured == {"ccd_revision": "ccd-revision", "ccd_snapshot_path": None}


@pytest.mark.parametrize(
    ("start", "invalid", "expected_calls"),
    (
        (1, {1}, [1, 2]),
        (2, {2}, [2, 0]),
    ),
)
def test_structure_dataset_retries_unresolved_crops_by_logical_index(
    monkeypatch: pytest.MonkeyPatch,
    start: int,
    invalid: set[int],
    expected_calls: list[int],
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index in invalid:
            raise UnresolvedTrainingCropError(f"record {index} is unresolved")
        return index

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[start] == expected_calls[-1]
    assert calls == expected_calls


def test_structure_dataset_preserves_valid_negative_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []
    monkeypatch.setattr(dataset, "_process_record", lambda index, **kwargs: calls.append(index) or index)

    assert dataset[-1] == 2
    assert calls == [2]


def test_structure_dataset_exhausts_each_logical_index_once_for_unresolved_crops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def unresolved(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise UnresolvedTrainingCropError(f"record {index} is unresolved")

    monkeypatch.setattr(dataset, "_process_record", unresolved)

    with pytest.raises(UnresolvedTrainingCropError) as exc_info:
        dataset[1]
    assert type(exc_info.value) is UnresolvedTrainingCropError
    assert calls == [1, 2, 0]


def test_structure_dataset_does_not_retry_ordinary_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif"]
    calls: list[int] = []

    def invalid(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise ValueError("ordinary processor failure")

    monkeypatch.setattr(dataset, "_process_record", invalid)

    with pytest.raises(ValueError, match="ordinary processor failure") as exc_info:
        dataset[0]
    assert type(exc_info.value) is ValueError
    assert calls == [0]


@pytest.mark.parametrize("payload", ["[]", "null", '"not an object"'])
def test_structure_dataset_rejects_non_mapping_index(tmp_path: Path, payload: str) -> None:
    structure_path = tmp_path / "sample.cif"
    structure_path.touch()
    index_path = tmp_path / "index.json"
    index_path.write_text(payload, encoding="utf-8")
    msa_path = tmp_path / "msa"
    msa_path.mkdir()
    with pytest.raises(ValueError, match="top level"):
        StructureDataset(structure_path, lambda value: value, seq_to_pdb_index=index_path, msa_path=msa_path)


def test_runner_routes_indexed_msa_without_leaking_control_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def structure_dataset(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    split = SimpleNamespace(cluster_sampling=SimpleNamespace(enabled=False), source="train.cif")
    split.items = lambda: {
        "source": "train.cif",
        "seq_to_pdb_index": "index.json",
        "msa_path": "msa",
        "max_tokens": 64,
    }.items()
    split.get = lambda key, default=None: dict(split.items()).get(key, default)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        config=SimpleNamespace(data={"train": split}),
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["seq_to_pdb_index"] == "index.json"
    assert captured["msa_path"] == "msa"
    assert captured["processor_kwargs"] == {"max_tokens": 64}


def test_runner_routes_indexed_msa_to_cluster_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def clustered_dataset(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.runners.dataset.ClusteredStructureDataset", clustered_dataset)
    monkeypatch.setattr(
        "ofoldx.runners.cluster.default_cluster_indices_path",
        lambda *args, **kwargs: pytest.fail("default cache path used despite explicit indices_file"),
    )

    class Split(dict[str, Any]):
        source = "train.cif"
        cluster_sampling = SimpleNamespace(
            enabled=True,
            cluster_file="clusters.txt",
            indices_file="explicit-indices.jsonl",
            contact_cutoff=5.0,
            include_all_chain_pairs=False,
            exclude_mol_types=("ions",),
            beta={"chain": 0.5, "interface": 1.0},
            alpha={"prot": 3.0, "nuc": 3.0, "ligand": 1.0},
        )

    split = Split(
        source="train.cif",
        seq_to_pdb_index="index.json",
        msa_path="msa",
        cluster_sampling=Split.cluster_sampling,
    )
    config = {"data": {"train": split}}
    config_object = SimpleNamespace(data=config["data"], get=config.get)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        _forward_cluster_indices=True,
        config=config_object,
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["seq_to_pdb_index"] == "index.json"
    assert captured["msa_path"] == "msa"
    assert captured["cluster_indices_file"] == "explicit-indices.jsonl"
    assert captured["processor_kwargs"] == {}
