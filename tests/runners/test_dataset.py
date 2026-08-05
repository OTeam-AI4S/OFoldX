from __future__ import annotations

import importlib.util
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from torch.utils.data import DataLoader

from ofoldx.io.structure import NoRetainedStructureError
from ofoldx.processor import UnresolvedTrainingCropError
from ofoldx.runners.dataset import (
    _PROCESSING_STAGE,
    ConsecutiveDataLoadingErrors,
    ProcessingTimeoutExhaustionError,
    StructureDataset,
    StructureProcessingTimeoutError,
)
from ofoldx.runners.runner import Runner


def _identity(value: Any) -> Any:
    return value


def test_structure_dataset_discovers_compound_gzip_suffixes(tmp_path: Path) -> None:
    structure_paths = [
        tmp_path / "first.cif.gz",
        tmp_path / "second.mmcif.gz",
        tmp_path / "third.pdb.gz",
    ]
    unsupported = tmp_path / "manifest.txt.gz"
    for path in [*structure_paths, unsupported]:
        path.touch()
    expected = sorted(str(path) for path in structure_paths)

    assert StructureDataset(structure_paths[0], _identity).records == [str(structure_paths[0])]
    assert StructureDataset(tmp_path, _identity).records == expected
    assert StructureDataset(str(tmp_path / "*.gz"), _identity).records == expected
    with pytest.raises(ValueError, match="gzip-compressed dataset source"):
        StructureDataset(unsupported, _identity)


class _PersistentWorkerTimeoutDataset(StructureDataset):
    def __init__(self) -> None:
        super().__init__(None, _identity, processing_timeout_seconds=0.02)
        self.records = ["slow.cif", "success.cif"]

    def _process_record(self, index: int, **extra_processor_kwargs: Any) -> str:
        if index == 0:
            time.sleep(1.0)
        return f"{index}:{os.getpid()}"


class _TwoWorkerRecoveryDataset(StructureDataset):
    def __init__(self, training_log_file: Path) -> None:
        super().__init__(
            None,
            _identity,
            max_consecutive_processing_errors=5,
            training_log_file=training_log_file,
        )
        self.records = ["bad.cif", "success.cif"]

    def _process_record(self, index: int, **extra_processor_kwargs: Any) -> int:
        if index == 0:
            raise UnresolvedTrainingCropError("worker-local malformed sample")
        return index


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


def test_structure_dataset_retries_no_retained_structure_and_wraps_by_logical_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index == 2:
            raise NoRetainedStructureError("record 2 has no retained structure")
        return index

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[2] == 0
    assert calls == [2, 0]


def test_structure_dataset_exhausts_no_retained_structures_once_and_preserves_last_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(None, lambda value: value)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def no_retained_structure(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise NoRetainedStructureError(f"record {index} has no retained structure")

    monkeypatch.setattr(dataset, "_process_record", no_retained_structure)

    with pytest.raises(UnresolvedTrainingCropError) as exc_info:
        dataset[1]
    assert type(exc_info.value) is UnresolvedTrainingCropError
    assert calls == [1, 2, 0]
    assert type(exc_info.value.__cause__) is NoRetainedStructureError
    assert str(exc_info.value.__cause__) == "record 0 has no retained structure"


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


@pytest.mark.parametrize("failure_count", range(1, 5))
def test_structure_dataset_opt_in_recovers_from_up_to_four_consecutive_data_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_count: int,
) -> None:
    dataset = StructureDataset(None, lambda value: value, max_consecutive_processing_errors=5)
    dataset.records = [f"record-{index}.cif" for index in range(6)]
    calls: list[int] = []

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index < failure_count:
            raise UnresolvedTrainingCropError(f"bad record {index}")
        return index

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[0] == failure_count
    assert calls == list(range(failure_count + 1))


def test_structure_dataset_fifth_consecutive_data_error_is_terminal_and_preserves_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(
        None,
        lambda value: value,
        max_consecutive_processing_errors=5,
        training_log_file=tmp_path / "20260804-221122.log",
    )
    dataset.records = [f"record-{index}.cif" for index in range(6)]
    calls: list[int] = []

    def fail(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise UnresolvedTrainingCropError(f"bad record {index}")

    monkeypatch.setattr(dataset, "_process_record", fail)

    with pytest.raises(ConsecutiveDataLoadingErrors) as exc_info:
        dataset[0]
    assert calls == [0, 1, 2, 3, 4]
    assert type(exc_info.value.__cause__) is UnresolvedTrainingCropError
    assert str(exc_info.value.__cause__) == "bad record 4"
    sidecar = next(tmp_path.glob("20260804-221122.data-errors.*.jsonl"))
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert [row["streak"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["action"] for row in rows] == [
        "skip_to_next_candidate",
        "skip_to_next_candidate",
        "skip_to_next_candidate",
        "skip_to_next_candidate",
        "raise_consecutive_error_limit",
    ]


def test_structure_dataset_short_dataset_exhausts_each_unique_candidate_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = StructureDataset(
        None,
        lambda value: value,
        max_consecutive_processing_errors=5,
        training_log_file=tmp_path / "20260804-221122.log",
    )
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def fail(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise UnresolvedTrainingCropError(f"bad record {index}")

    monkeypatch.setattr(dataset, "_process_record", fail)

    with pytest.raises(ConsecutiveDataLoadingErrors) as exc_info:
        dataset[1]
    assert calls == [1, 2, 0]
    assert type(exc_info.value.__cause__) is UnresolvedTrainingCropError
    assert str(exc_info.value.__cause__) == "bad record 0"
    assert "3" in str(exc_info.value)
    sidecar = next(tmp_path.glob("20260804-221122.data-errors.*.jsonl"))
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert [row["streak"] for row in rows] == [1, 2, 3]
    assert rows[-1]["action"] == "raise_candidate_exhaustion"


def test_structure_dataset_success_resets_the_next_request_streak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_log = tmp_path / "20260804-221122.log"
    dataset = StructureDataset(
        None,
        lambda value: value,
        max_consecutive_processing_errors=5,
        training_log_file=training_log,
    )
    dataset.records = ["bad.cif", "good.cif"]
    calls: list[int] = []

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index == 0:
            raise UnresolvedTrainingCropError("bad record")
        return index

    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[0] == 1
    assert dataset[0] == 1
    assert calls == [0, 1, 0, 1]
    sidecar = next(tmp_path.glob("20260804-221122.data-errors.*.jsonl"))
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert [row["streak"] for row in rows] == [1, 1]


def test_structure_dataset_logs_duplicate_atom_value_error_with_complete_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad_path = tmp_path / "duplicate.cif"
    good_path = tmp_path / "good.cif"
    bad_path.touch()
    good_path.touch()
    training_log = tmp_path / "20260804-221122.log"
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[]))
    message = "duplicate source atom declaration: (None, 'B', 91, '91', None, 'LYS', 'C', None)"

    def read_structure(path: str, **kwargs: Any) -> object:
        if path == str(bad_path):
            raise ValueError(message)
        return structure

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    monkeypatch.setattr("ofoldx.runners.dataset.get_worker_info", lambda: SimpleNamespace(id=3))
    monkeypatch.setenv("RANK", "45")
    dataset = StructureDataset(
        None,
        lambda value: "success",
        max_consecutive_processing_errors=5,
        training_log_file=training_log,
    )
    dataset.records = [str(bad_path), str(good_path)]

    with caplog.at_level(logging.WARNING, logger="ofoldx.runners.dataset"):
        assert dataset[0] == "success"

    sidecar = tmp_path / "20260804-221122.data-errors.rank-00045.worker-03.jsonl"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["stage"] == "read_structure"
    assert payload["error_type"] == "ValueError"
    assert payload["error_message"] == message
    assert "ValueError" in payload["traceback"] and message in payload["traceback"]
    assert payload["logical_index"] == 0
    assert payload["backing_record_index"] == 0
    assert payload["structure_path"] == str(bad_path)
    assert payload["rank"] == 45
    assert payload["worker_id"] == 3
    assert payload["streak"] == 1
    assert payload["threshold"] == 5
    assert payload["action"] == "skip_to_next_candidate"
    assert payload["elapsed_seconds"] >= 0
    assert isinstance(payload["pid"], int)
    assert payload["timestamp"]
    assert any(message in record.getMessage() for record in caplog.records)


def test_structure_dataset_sidecar_recovers_from_interrupted_and_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_log = tmp_path / "20260804-221122.log"
    dataset = StructureDataset(None, _identity, training_log_file=training_log)
    real_write = os.write
    calls = 0

    def flaky_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("interrupted append")
        if calls == 2:
            partial = max(1, len(payload) // 2)
            return real_write(descriptor, payload[:partial])
        return real_write(descriptor, payload)

    monkeypatch.setattr("ofoldx.runners.dataset.os.write", flaky_write)
    event = {"stage": "read_structure", "error_type": "ValueError", "error_message": "bad atom row"}
    dataset._append_data_error_event(event, rank_component="00012", worker_id=4)

    sidecar = tmp_path / "20260804-221122.data-errors.rank-00012.worker-04.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert calls >= 3
    assert len(lines) == 1
    assert json.loads(lines[0]) == event


def test_structure_dataset_two_workers_write_separate_valid_error_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_log = tmp_path / "20260804-221122.log"
    monkeypatch.setenv("RANK", "7")
    loader = DataLoader(
        _TwoWorkerRecoveryDataset(training_log),
        batch_size=None,
        sampler=[0, 0],
        num_workers=2,
    )

    assert list(loader) == [1, 1]
    sidecars = sorted(tmp_path.glob("20260804-221122.data-errors.rank-00007.worker-*.jsonl"))
    assert [path.name.rsplit("worker-", 1)[1] for path in sidecars] == ["00.jsonl", "01.jsonl"]
    for worker_id, sidecar in enumerate(sidecars):
        lines = sidecar.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["worker_id"] == worker_id
        assert event["rank"] == 7
        assert event["streak"] == 1
        assert event["action"] == "skip_to_next_candidate"


def test_structure_dataset_counts_mixed_recoverable_failures_in_one_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[Exception] = [
        StructureProcessingTimeoutError("slow structure", stage="read_structure"),
        UnresolvedTrainingCropError("empty crop"),
        NoRetainedStructureError("no retained atoms"),
        ValueError("malformed atom row"),
    ]
    structures = [SimpleNamespace(index=index, system=SimpleNamespace(entity_type=[])) for index in range(5)]
    dataset = StructureDataset(None, lambda value: value, max_consecutive_processing_errors=5)
    dataset.records = [f"record-{index}.cif" for index in range(5)]
    calls: list[int] = []

    def read_structure(path: str, **kwargs: Any) -> object:
        return structures[int(path.removeprefix("record-").removesuffix(".cif"))]

    def process(structure: Any, **kwargs: Any) -> int:
        calls.append(structure.index)
        if structure.index < len(errors):
            raise errors[structure.index]
        return structure.index

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    dataset.processor = process

    assert dataset[0] == 4
    assert calls == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "error",
    [
        TypeError("signature bug"),
        RuntimeError("framework failure"),
        TimeoutError("unrelated timeout"),
        MemoryError("host out of memory"),
        KeyboardInterrupt(),
    ],
    ids=["type-error", "runtime-error", "unrelated-timeout", "memory-error", "keyboard-interrupt"],
)
def test_structure_dataset_never_retries_non_data_or_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[]))
    calls: list[object] = []

    def fail_processor(value: object, **kwargs: Any) -> None:
        calls.append(value)
        raise error

    dataset = StructureDataset(None, fail_processor, max_consecutive_processing_errors=5)
    dataset.records = ["zero.cif", "one.cif"]
    monkeypatch.setattr("ofoldx.io.read_structure", lambda path, **kwargs: structure)

    with pytest.raises(type(error), match=str(error) or None):
        dataset[0]
    assert calls == [structure]


@pytest.mark.parametrize("threshold", [True, False, 0, -1, 1.5, "5"])
def test_structure_dataset_rejects_invalid_consecutive_error_threshold(threshold: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_consecutive_processing_errors"):
        StructureDataset(None, lambda value: value, max_consecutive_processing_errors=threshold)


def _require_posix_processing_timer() -> None:
    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if os.name != "posix" or any(not hasattr(signal, name) for name in required):
        pytest.skip("POSIX interval timers are unavailable")


def test_structure_dataset_short_timeout_logs_provenance_and_uses_successor(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _require_posix_processing_timer()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if any(previous_timer):
        pytest.skip("test process already has an active ITIMER_REAL timer")
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=0.02)
    dataset.records = ["slow.cif", "success.cif"]
    calls: list[int] = []

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index == 0:
            time.sleep(1.0)
        return index

    monkeypatch.setattr(dataset, "_process_record", process)
    monkeypatch.setattr("ofoldx.runners.dataset.get_worker_info", lambda: SimpleNamespace(id=4))
    monkeypatch.setenv("RANK", "7")
    with caplog.at_level(logging.WARNING, logger="ofoldx.runners.dataset"):
        assert dataset[0] == 1

    assert calls == [0, 1]
    assert len(caplog.messages) == 1
    message = caplog.messages[0]
    assert "StructureProcessingTimeoutError" in message
    assert "actual seconds" in message
    assert "configured deadline 0.02 seconds" in message
    assert "stage unknown" in message
    assert "logical index 0" in message
    assert "backing record index 0/path 'slow.cif'" in message
    assert "rank 7" in message and "DataLoader worker ID 4" in message
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


def test_structure_dataset_timeout_exhaustion_is_bounded_and_preserves_last_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _require_posix_processing_timer()
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.0)
    dataset.records = ["zero.cif", "one.cif", "two.cif"]
    calls: list[int] = []

    def timeout(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise StructureProcessingTimeoutError(f"timed out record {index}")

    monkeypatch.setattr(dataset, "_process_record", timeout)
    with caplog.at_level(logging.WARNING, logger="ofoldx.runners.dataset"):
        with pytest.raises(ProcessingTimeoutExhaustionError) as exc_info:
            dataset[1]

    assert calls == [1, 2, 0]
    assert len(caplog.messages) == 3
    assert type(exc_info.value.__cause__) is StructureProcessingTimeoutError
    assert str(exc_info.value.__cause__) == "timed out record 0"
    assert "scanning 3 logical candidates and processing 3 attempts" in str(exc_info.value)
    assert exc_info.value.had_processing_timeout is True
    assert exc_info.value.timeout_count == 3
    assert exc_info.value.scanned_candidates == 3
    assert exc_info.value.processed_attempts == 3


@pytest.mark.parametrize(
    ("failing_stage", "manifest_record"),
    [("read_structure", False), ("read_msa", True), ("processor", False)],
)
def test_structure_dataset_timeout_records_active_stage_and_restores_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
    manifest_record: bool,
) -> None:
    structure_path = tmp_path / "sample.cif"
    structure_path.touch()
    source: Path = structure_path
    if manifest_record:
        msa_path = tmp_path / "sample.a3m"
        msa_path.touch()
        source = tmp_path / "manifest.jsonl"
        source.write_text(
            json.dumps({"structure": structure_path.name, "msa": {"path": msa_path.name, "entity_id": "1"}}),
            encoding="utf-8",
        )
    structure = SimpleNamespace(system=SimpleNamespace(entity_type=[1]))

    def read_structure(path: str, **kwargs: Any) -> object:
        if failing_stage == "read_structure":
            raise StructureProcessingTimeoutError("stage timeout")
        return structure

    def read_msa(*args: Any, **kwargs: Any) -> object:
        if failing_stage == "read_msa":
            raise StructureProcessingTimeoutError("stage timeout")
        return object()

    def processor(value: object, **kwargs: Any) -> object:
        if failing_stage == "processor":
            raise StructureProcessingTimeoutError("stage timeout")
        return value

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    monkeypatch.setattr("ofoldx.io.read_msa", read_msa)
    dataset = StructureDataset(source, processor)

    assert _PROCESSING_STAGE.get() == "unknown"
    with pytest.raises(ProcessingTimeoutExhaustionError) as exc_info:
        dataset[0]
    assert isinstance(exc_info.value.__cause__, StructureProcessingTimeoutError)
    assert exc_info.value.__cause__.stage == failing_stage
    assert _PROCESSING_STAGE.get() == "unknown"


def test_structure_dataset_timeout_fallback_reuses_persistent_dataloader_worker() -> None:
    _require_posix_processing_timer()
    dataset = _PersistentWorkerTimeoutDataset()
    loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=[0],
        num_workers=1,
        persistent_workers=True,
    )
    try:
        first_epoch = list(loader)
        second_epoch = list(loader)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert len(first_epoch) == len(second_epoch) == 1
    first_index, first_pid = first_epoch[0].split(":")
    second_index, second_pid = second_epoch[0].split(":")
    assert first_index == second_index == "1"
    assert first_pid == second_pid
    assert int(first_pid) != os.getpid()


def test_structure_dataset_arms_a_fresh_timer_for_each_timeout_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_posix_processing_timer()
    if any(signal.getitimer(signal.ITIMER_REAL)):
        pytest.skip("test process already has an active ITIMER_REAL timer")
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.25)
    dataset.records = ["slow.cif", "success.cif"]
    calls: list[int] = []
    timer_calls: list[tuple[int, float, float]] = []

    def setitimer(which: int, seconds: float, interval: float = 0.0) -> tuple[float, float]:
        timer_calls.append((which, seconds, interval))
        return (0.0, 0.0)

    def process(index: int, **kwargs: Any) -> int:
        calls.append(index)
        if index == 0:
            raise StructureProcessingTimeoutError("first attempt")
        return index

    monkeypatch.setattr(signal, "setitimer", setitimer)
    monkeypatch.setattr(dataset, "_process_record", process)

    assert dataset[0] == 1
    assert calls == [0, 1]
    assert [call for call in timer_calls if call[1] == 1.25] == [
        (signal.ITIMER_REAL, 1.25, 0.0),
        (signal.ITIMER_REAL, 1.25, 0.0),
    ]


@pytest.mark.parametrize(
    "error",
    [ValueError("ordinary failure"), TimeoutError("unrelated timeout")],
    ids=["value-error", "unrelated-timeout-error"],
)
def test_structure_dataset_deadline_cleanup_does_not_retry_unrelated_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    _require_posix_processing_timer()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if any(previous_timer):
        pytest.skip("test process already has an active ITIMER_REAL timer")
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.0)
    dataset.records = ["zero.cif", "one.cif"]
    calls: list[int] = []

    def fail(index: int, **kwargs: Any) -> None:
        calls.append(index)
        raise error

    monkeypatch.setattr(dataset, "_process_record", fail)

    with pytest.raises(type(error), match=str(error)):
        dataset[0]
    assert calls == [0]
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


@pytest.mark.parametrize(
    "timeout",
    [True, False, 0, -1, float("nan"), float("inf"), float("-inf"), "not-a-number"],
)
def test_structure_dataset_rejects_invalid_processing_timeout(timeout: Any) -> None:
    with pytest.raises(ValueError, match="processing_timeout_seconds.*finite positive"):
        StructureDataset(None, lambda value: value, processing_timeout_seconds=timeout)


def test_structure_dataset_timeout_requires_main_thread() -> None:
    _require_posix_processing_timer()
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.0)
    dataset.records = ["zero.cif"]
    errors: list[BaseException] = []

    def load() -> None:
        try:
            dataset[0]
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=load)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])


def test_structure_dataset_timeout_refuses_active_real_timer() -> None:
    _require_posix_processing_timer()
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if any(previous_timer):
        pytest.skip("test process already has an active ITIMER_REAL timer")
    dataset = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.0)
    dataset.records = ["zero.cif"]
    signal.setitimer(signal.ITIMER_REAL, 60.0)
    try:
        with pytest.raises(RuntimeError, match="ITIMER_REAL timer is already active"):
            dataset[0]
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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
        "processing_timeout_seconds": 2.5,
        "max_tokens": 64,
    }.items()
    split.get = lambda key, default=None: dict(split.items()).get(key, default)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        config=SimpleNamespace(data={"train": split}, get=lambda key, default=None: default),
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["seq_to_pdb_index"] == "index.json"
    assert captured["msa_path"] == "msa"
    assert captured["processing_timeout_seconds"] == 2.5
    assert captured["processor_kwargs"] == {"max_tokens": 64}


def test_runner_enables_default_processing_timeout_for_ordinary_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def structure_dataset(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    split = SimpleNamespace(cluster_sampling=SimpleNamespace(enabled=False), source="train.cif")
    split.items = lambda: {"source": "train.cif", "processor_seed": 9}.items()
    split.get = lambda key, default=None: dict(split.items()).get(key, default)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        config=SimpleNamespace(data={"train": split}, get=lambda key, default=None: default),
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["processing_timeout_seconds"] == 30.0
    assert captured["processor_kwargs"] == {"processor_seed": 9}


def test_runner_disables_default_processing_timeout_for_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def structure_dataset(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    split = SimpleNamespace(cluster_sampling=SimpleNamespace(enabled=False), source="val.cif")
    split.items = lambda: {"source": "val.cif", "processor_seed": 9}.items()
    split.get = lambda key, default=None: dict(split.items()).get(key, default)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        config=SimpleNamespace(data={"val": split}, get=lambda key, default=None: default),
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["processing_timeout_seconds"] is None
    assert captured["processor_kwargs"] == {"processor_seed": 9}


def test_runner_allows_explicit_processing_timeout_for_evaluation_without_forwarding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def structure_dataset(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    split = SimpleNamespace(cluster_sampling=SimpleNamespace(enabled=False), source="val.cif")
    split.items = lambda: {
        "source": "val.cif",
        "processing_timeout_seconds": 4.5,
        "processor_seed": 9,
    }.items()
    split.get = lambda key, default=None: dict(split.items()).get(key, default)
    fake_runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        config=SimpleNamespace(data={"val": split}, get=lambda key, default=None: default),
    )

    Runner.build_datasets(fake_runner)  # type: ignore[arg-type]
    assert captured["processing_timeout_seconds"] == 4.5
    assert captured["processor_kwargs"] == {"processor_seed": 9}


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
    assert captured["processing_timeout_seconds"] == 30.0
    assert captured["processor_kwargs"] == {}
