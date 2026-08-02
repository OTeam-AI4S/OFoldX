from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import ofoldx.runners.train as canonical


def _load_wrapper() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("ofoldx_test_train_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compatibility_wrapper_reexports_canonical_cli() -> None:
    wrapper = _load_wrapper()

    assert wrapper.parse_args is canonical.parse_args
    assert wrapper.build_config is canonical.build_config
    assert wrapper.main is canonical.main


@pytest.mark.parametrize(
    ("flag", "key", "value"),
    [
        ("--ccd-revision", "ccd_revision", "ccd-v1"),
        ("--ccd-snapshot-path", "ccd_snapshot_path", "/data/ccd"),
    ],
)
def test_explicit_ccd_locator_maps_to_a_copied_processor_config(
    flag: str,
    key: str,
    value: str,
) -> None:
    args = canonical.parse_args(["train.cif", "--processor-config", '{"feature": 1}', flag, value])
    original = args.processor_config

    config = canonical.build_config(args)

    assert config["processor"] == {"feature": 1, key: value}
    assert config["processor"] is not original
    assert original == {"feature": 1}


def test_default_processor_config_does_not_override_reader_ingestion_policy() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif"]))

    assert config["processor"] == {}
    assert "include_hydrogens" not in config["processor"]
    assert "include_water" not in config["processor"]
    assert "bond_policy" not in config["processor"]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], False), (["--gradient-checkpointing"], True), (["--no-gradient-checkpointing"], False)],
)
def test_gradient_checkpointing_cli_maps_to_top_level_config(flags: list[str], expected: bool) -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", *flags]))

    assert config["gradient_checkpointing"] is expected


def test_gradient_checkpointing_is_independent_of_training_state_checkpointing() -> None:
    config = canonical.build_config(
        canonical.parse_args(["train.cif", "--gradient-checkpointing", "--no-checkpoint"])
    )

    assert config["gradient_checkpointing"] is True
    assert config["ckpt"]["enabled"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["train.cif", "--ccd-revision", "one", "--ccd-snapshot-path", "two"],
        ["train.cif", "--ccd-revision", "one", "--processor-config", '{"ccd_revision": null}'],
        ["train.cif", "--ccd-revision", "one", "--processor-config", '{"ccd_snapshot_path": null}'],
        ["train.cif", "--pretrained", "repo", "--ccd-revision", "one"],
        ["train.cif", "--pretrained", "repo", "--processor-config", '{"feature": 1}'],
        [
            "train.cif",
            "--processor-config",
            '{"ccd_revision": "one", "ccd_snapshot_path": "two"}',
        ],
        ["train.cif", "--ccd-revision", ""],
        ["train.cif", "--ccd-snapshot-path", "   "],
        ["train.cif", "--processor-config", '{"ccd_revision": ""}'],
        ["train.cif", "--processor-config", '{"ccd_snapshot_path": "   "}'],
    ],
)
def test_ccd_locator_conflicts_and_empty_values_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(argv)


def test_pretrained_revision_remains_distinct_from_ccd_revision() -> None:
    args = canonical.parse_args(["train.cif", "--pretrained", "repo", "--revision", "model-rev"])
    config = canonical.build_config(args)

    assert config["model"]["revision"] == "model-rev"
    assert config["processor"] == {}


def test_print_config_does_not_construct_runner(capsys: pytest.CaptureFixture[str]) -> None:
    assert canonical.main(["train.cif", "--ccd-revision", "ccd-v1", "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["processor"] == {"ccd_revision": "ccd-v1"}


def test_legacy_logging_config_is_exact_and_new_backends_are_absent_by_default() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", "--no-logging", "--log-interval", "7"]))

    assert config["logging"] == {"enabled": False, "interval": 7}
    assert "tensorboard" not in config
    assert "wandb" not in config


def test_tensorboard_default_and_explicit_config_mapping() -> None:
    default = canonical.build_config(canonical.parse_args(["train.cif", "--tensorboard"]))
    explicit = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--tensorboard",
                "--tensorboard-log-dir",
                "events/custom",
                "--tensorboard-flush-secs",
                "15",
            ]
        )
    )

    assert default["tensorboard"] == {"enabled": True}
    assert explicit["tensorboard"] == {
        "enabled": True,
        "log_dir": "events/custom",
        "flush_secs": 15,
    }


def test_wandb_full_config_mapping_preserves_tag_order() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--wandb",
                "--wandb-mode",
                "offline",
                "--wandb-project",
                "folding",
                "--wandb-entity",
                "research",
                "--wandb-id",
                "run-123",
                "--wandb-name",
                "baseline",
                "--wandb-group",
                "ablation",
                "--wandb-resume",
                "allow",
                "--wandb-tag",
                "first",
                "--wandb-tag",
                "second",
            ]
        )
    )

    assert config["wandb"] == {
        "enabled": True,
        "mode": "offline",
        "project": "folding",
        "entity": "research",
        "id": "run-123",
        "name": "baseline",
        "group": "ablation",
        "resume": "allow",
        "tags": ["first", "second"],
    }


def test_logging_backends_are_independent_and_combinable() -> None:
    config = canonical.build_config(
        canonical.parse_args(["train.cif", "--no-logging", "--tensorboard", "--wandb"])
    )

    assert config["logging"] == {"enabled": False, "interval": None}
    assert config["tensorboard"] == {"enabled": True}
    assert config["wandb"] == {"enabled": True}


@pytest.mark.parametrize(
    "backend_arguments",
    [
        ["--tensorboard-log-dir", "events"],
        ["--tensorboard-flush-secs", "10"],
        ["--no-tensorboard", "--tensorboard-log-dir", "events"],
        ["--wandb-mode", "offline"],
        ["--wandb-project", "project"],
        ["--wandb-entity", "entity"],
        ["--wandb-id", "id"],
        ["--wandb-name", "name"],
        ["--wandb-group", "group"],
        ["--wandb-resume", "auto"],
        ["--wandb-tag", "tag"],
        ["--no-wandb", "--wandb-mode", "offline"],
    ],
)
def test_backend_specific_options_require_enabled_backend(backend_arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *backend_arguments])


@pytest.mark.parametrize(
    "backend_arguments",
    [
        ["--tensorboard", "--tensorboard-log-dir", ""],
        ["--tensorboard", "--tensorboard-log-dir", "   "],
        ["--tensorboard", "--tensorboard-flush-secs", "0"],
        ["--tensorboard", "--tensorboard-flush-secs", "-1"],
        ["--wandb", "--wandb-mode", "invalid"],
        ["--wandb", "--wandb-project", ""],
        ["--wandb", "--wandb-entity", "   "],
        ["--wandb", "--wandb-id", ""],
        ["--wandb", "--wandb-name", ""],
        ["--wandb", "--wandb-group", ""],
        ["--wandb", "--wandb-resume", "invalid"],
        ["--wandb", "--wandb-tag", "   "],
    ],
)
def test_invalid_backend_values_are_rejected(backend_arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *backend_arguments])


def test_print_config_with_optional_backends_does_not_import_optional_logger_packages(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("tensorboard", "wandb")):
            raise AssertionError(f"unexpected optional logger import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert canonical.main(
        ["train.cif", "--tensorboard", "--wandb", "--wandb-mode", "offline", "--print-config"]
    ) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["tensorboard"] == {"enabled": True}
    assert config["wandb"] == {"enabled": True, "mode": "offline"}


def test_slurm_launcher_uses_job_name_and_job_id_log_paths() -> None:
    launcher = (Path(__file__).parents[2] / "ofoldx" / "runners" / "submit_train.slurm").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --output=/data/users/qinghanw313/AlloDesign/logs/%x-%j.out" in launcher
    assert "#SBATCH --error=/data/users/qinghanw313/AlloDesign/logs/%x-%j.err" in launcher


def test_crop_settings_are_absent_by_default_from_train_and_val() -> None:
    config = canonical.build_config(canonical.parse_args(["train.cif", "--val-source", "val.cif"]))

    assert "max_tokens" not in config["data"]["train"]
    assert "crop_method_weights" not in config["data"]["train"]
    assert "max_tokens" not in config["data"]["val"]
    assert "crop_method_weights" not in config["data"]["val"]


def test_max_tokens_uses_continuous_crop_without_cluster_sampling() -> None:
    args = canonical.parse_args(["train.cif", "--max-tokens", "384"])
    config = canonical.build_config(args)

    assert args.crop_method_weights == [1.0, 0.0, 0.0]
    assert config["data"]["train"]["max_tokens"] == 384
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


def test_max_tokens_uses_mixed_crop_with_cluster_sampling() -> None:
    args = canonical.parse_args(
        ["train.cif", "--max-tokens", "384", "--cluster-file", "clusters.txt"]
    )
    config = canonical.build_config(args)

    assert args.crop_method_weights == [0.2, 0.4, 0.4]
    assert config["data"]["train"]["crop_method_weights"] == [0.2, 0.4, 0.4]


def test_explicit_continuous_weights_allow_explicit_protenix_without_cluster_sampling() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--model",
            "protenix",
            "--max-tokens",
            "384",
            "--crop-method-weights",
            "2",
            "0",
            "0",
        ]
    )
    config = canonical.build_config(args)

    assert config["data"]["train"]["crop_method_weights"] == [2.0, 0.0, 0.0]


def test_explicit_crop_weights_map_only_to_training_split() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--val-source",
            "val.cif",
            "--eval-source",
            "benchmark=test.cif",
            "--max-tokens",
            "256",
            "--crop-method-weights",
            "0.5",
            "0.25",
            "0.25",
            "--cluster-file",
            "clusters.txt",
        ]
    )
    config = canonical.build_config(args)

    assert config["data"]["train"]["max_tokens"] == 256
    assert config["data"]["train"]["crop_method_weights"] == [0.5, 0.25, 0.25]
    assert config["data"]["val"] == {"source": "val.cif"}
    assert config["data"]["benchmark"] == {"source": "test.cif"}


def test_crop_size_alias_maps_to_max_tokens() -> None:
    args = canonical.parse_args(["train.cif", "--crop-size", "128"])
    config = canonical.build_config(args)

    assert args.max_tokens == 128
    assert config["data"]["train"]["max_tokens"] == 128


def test_pretrained_repository_is_allowed_when_crop_architecture_is_not_known_yet() -> None:
    args = canonical.parse_args(["train.cif", "--pretrained", "organization/model", "--max-tokens", "128"])
    config = canonical.build_config(args)

    assert config["model"]["pretrained"] == "organization/model"
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "argv",
    [
        ["train.cif", "--max-tokens", "0"],
        ["train.cif", "--max-tokens", "-1"],
        ["train.cif", "--crop-method-weights", "1", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "-1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "nan", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "inf", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "-inf", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "1", "0", "0", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "1", "0"],
        ["train.cif", "--max-tokens", "128", "--crop-method-weights", "0", "0", "1"],
        ["train.cif", "--model", "ligandmpnn", "--max-tokens", "128"],
    ],
)
def test_invalid_crop_options_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(argv)


def test_print_config_includes_normalized_crop_settings(capsys: pytest.CaptureFixture[str]) -> None:
    assert canonical.main(["train.cif", "--max-tokens", "192", "--print-config"]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["data"]["train"]["max_tokens"] == 192
    assert config["data"]["train"]["crop_method_weights"] == [1.0, 0.0, 0.0]


def test_multiple_named_evaluation_sources_preserve_order_and_select_first_score() -> None:
    args = canonical.parse_args(
        [
            "train.cif",
            "--eval-source",
            "validation=val.cif",
            "--eval-source",
            "benchmark=data/path=with=equals.cif",
        ]
    )

    config = canonical.build_config(args)

    assert args.eval_sources == [
        ("validation", "val.cif"),
        ("benchmark", "data/path=with=equals.cif"),
    ]
    assert config["evaluate_splits"] == ["validation", "benchmark"]
    assert list(config["data"]) == ["train", "validation", "benchmark"]
    assert config["data"]["validation"] == {"source": "val.cif"}
    assert config["data"]["benchmark"] == {"source": "data/path=with=equals.cif"}
    assert config["score"] == {"split": "validation", "metric": "loss"}


def test_val_source_coexists_with_named_evaluation_and_has_score_priority() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--eval-source",
                "benchmark=test.cif",
                "--val-source",
                "val.cif",
            ]
        )
    )

    assert config["evaluate_splits"] == ["val", "benchmark"]
    assert config["score"] == {"split": "val", "metric": "loss"}


def test_explicit_score_split_selects_one_configured_evaluation_split() -> None:
    config = canonical.build_config(
        canonical.parse_args(
            [
                "train.cif",
                "--val-source",
                "val.cif",
                "--eval-source",
                "benchmark=test.cif",
                "--score-split",
                "benchmark",
            ]
        )
    )

    assert config["score"] == {"split": "benchmark", "metric": "loss"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--eval-source", "missing-separator"],
        ["--eval-source", "=path.cif"],
        ["--eval-source", "1invalid=path.cif"],
        ["--eval-source", "invalid.name=path.cif"],
        ["--eval-source", "train=path.cif"],
        ["--eval-source", "empty="],
        ["--eval-source", "empty=   "],
        ["--eval-source", "dup=one.cif", "--eval-source", "dup=two.cif"],
        ["--val-source", "one.cif", "--eval-source", "val=two.cif"],
        ["--score-split", "val"],
        ["--eval-source", "validation=val.cif", "--score-split", "unknown"],
    ],
)
def test_invalid_named_evaluation_options_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", *arguments])


def test_main_checks_every_evaluation_split_and_closes_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"closed": False, "trained": False}

    class FakeModel:
        def set_backend(self, backend: str) -> None:
            assert backend == "cueq"

    class FakeRunner:
        rank = 1
        model = FakeModel()

        def __init__(self, config: dict[str, Any]) -> None:
            assert config["evaluate_splits"] == ["first", "empty"]
            self.datasets = {"train": [object()], "first": [object()], "empty": []}

        def unwrap(self, model: FakeModel) -> FakeModel:
            return model

        def train(self) -> None:
            state["trained"] = True

        def close(self) -> None:
            state["closed"] = True

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)

    with pytest.raises(ValueError, match="evaluation source 'empty'.*produced an empty dataset"):
        canonical.main(
            [
                "train.cif",
                "--eval-source",
                "first=first.cif",
                "--eval-source",
                "empty=empty.cif",
                "--device",
                "cpu",
            ]
        )

    assert state == {"closed": True, "trained": False}


def test_main_distributed_error_fails_fast_without_closing_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"closed": 0, "fail_fast": 0}

    class FailFastExit(BaseException):
        pass

    class FakeModel:
        def set_backend(self, backend: str) -> None:
            assert backend == "cueq"

    class FakeRunner:
        distributed = True
        rank = 1
        model = FakeModel()

        def __init__(self, config: dict[str, Any]) -> None:
            self.datasets = {"train": [object()]}

        def unwrap(self, model: FakeModel) -> FakeModel:
            return model

        def train(self) -> None:
            raise RuntimeError("loader failed")

        def close(self) -> None:
            state["closed"] += 1

    def fail_fast() -> None:
        state["fail_fast"] += 1
        raise FailFastExit

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)
    monkeypatch.setattr(canonical, "_distributed_fail_fast", fail_fast)

    with pytest.raises(FailFastExit):
        canonical.main(["train.cif", "--device", "cpu"])

    assert state == {"closed": 0, "fail_fast": 1}


def test_distributed_fail_fast_reports_flushes_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class ExitCalled(BaseException):
        pass

    class RecordingStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def write(self, value: str) -> int:
            events.append((self.name, value))
            return len(value)

        def flush(self) -> None:
            events.append((f"{self.name}_flush", None))

    def exit_process(status: int) -> None:
        events.append(("exit", status))
        raise ExitCalled

    monkeypatch.setattr(canonical.sys, "stdout", RecordingStream("stdout"))
    monkeypatch.setattr(canonical.sys, "stderr", RecordingStream("stderr"))
    monkeypatch.setattr(canonical.os, "_exit", exit_process)

    try:
        raise RuntimeError("loader failed")
    except RuntimeError:
        with pytest.raises(ExitCalled):
            canonical._distributed_fail_fast()

    stderr = "".join(value for name, value in events if name == "stderr" and isinstance(value, str))
    assert "RuntimeError: loader failed" in stderr
    assert events[-3:] == [("stdout_flush", None), ("stderr_flush", None), ("exit", 1)]


def test_main_success_closes_runner_once(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"closed": 0, "trained": 0}

    class FakeModel:
        def set_backend(self, backend: str) -> None:
            assert backend == "cueq"

    class FakeRunner:
        rank = 1
        model = FakeModel()

        def __init__(self, config: dict[str, Any]) -> None:
            self.datasets = {"train": [object()]}

        def unwrap(self, model: FakeModel) -> FakeModel:
            return model

        def train(self) -> None:
            state["trained"] += 1

        def close(self) -> None:
            state["closed"] += 1

    monkeypatch.setattr("ofoldx.runners.runner.Runner", FakeRunner)

    assert canonical.main(["train.cif", "--device", "cpu"]) == 0
    assert state == {"closed": 1, "trained": 1}
