from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ofoldx.runners.train as train
from ofoldx.runners import runner as runner_module
from ofoldx.runners.collate import collate_with_model_kwargs
from ofoldx.runners.config import RunnerConfig
from ofoldx.runners.runner import Runner


def _sentinel_collate(batch: list[Any]) -> dict[str, Any]:
    r"""Return a minimal DanLing batch without depending on model fixtures."""

    assert batch == ["item"]
    model_input = {"input": "features", "target": "truth"}
    return {"input": model_input, "target": model_input, "loss_normalizer": 7}


class _FakeModel:
    def __init__(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> _FakeModel:
        self.training = mode
        return self


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "diffusion.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_cli_maps_diffusion_batch_and_chunk_without_changing_dataloader_batch() -> None:
    config = train.build_config(
        train.parse_args(
            [
                "train.cif",
                "--batch-size",
                "3",
                "--diffusion-batch-size",
                "5",
                "--diffusion-chunk-size",
                "2",
            ]
        )
    )

    assert config["dataloader"]["batch_size"] == 3
    assert config["diffusion_batch_size"] == 5
    assert config["diffusion_chunk_size"] == 2


def test_default_cli_omits_diffusion_fields() -> None:
    config = train.build_config(train.parse_args(["train.cif"]))

    assert "diffusion_batch_size" not in config
    assert "diffusion_chunk_size" not in config


def test_yaml_maps_diffusion_batch_and_chunk(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "train_source: train.cif\n"
        "batch_size: 3\n"
        "diffusion_batch_size: 5\n"
        "diffusion_chunk_size: 2\n",
    )

    config = train.build_config(train.parse_args(["--config", str(path)]))

    assert config["dataloader"]["batch_size"] == 3
    assert config["diffusion_batch_size"] == 5
    assert config["diffusion_chunk_size"] == 2


def test_yaml_null_diffusion_fields_are_equivalent_to_absence(tmp_path: Path) -> None:
    absent = train.build_config(
        train.parse_args(["--config", str(_write_config(tmp_path, "train_source: train.cif\n"))])
    )
    null_path = tmp_path / "null.yaml"
    null_path.write_text(
        "train_source: train.cif\n"
        "diffusion_batch_size: null\n"
        "diffusion_chunk_size: null\n",
        encoding="utf-8",
    )
    explicit_null = train.build_config(train.parse_args(["--config", str(null_path)]))

    assert explicit_null == absent
    assert "diffusion_batch_size" not in explicit_null
    assert "diffusion_chunk_size" not in explicit_null


@pytest.mark.parametrize("flag", ["--diffusion-batch-size", "--diffusion-chunk-size"])
@pytest.mark.parametrize("value", ["true", "0", "-1", "1.5", "two"])
def test_cli_rejects_invalid_diffusion_sizes(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        train.parse_args(["train.cif", flag, value])

    assert error.value.code == 2


@pytest.mark.parametrize("key", ["diffusion_batch_size", "diffusion_chunk_size"])
@pytest.mark.parametrize("rendered", ["true", "0", "-1", "1.5", '"2"'])
def test_yaml_rejects_invalid_diffusion_sizes(tmp_path: Path, key: str, rendered: str) -> None:
    path = tmp_path / f"{key}-{rendered.replace('/', '_')}.yaml"
    path.write_text(f"train_source: train.cif\n{key}: {rendered}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        train.parse_args(["--config", str(path)])

    assert error.value.code == 2


@pytest.mark.parametrize("key", ["diffusion_batch_size", "diffusion_chunk_size"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_runner_config_rejects_invalid_diffusion_sizes(key: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match=key):
        RunnerConfig({"model": {"name": "protenix"}, key: value})


@pytest.mark.parametrize("key", ["diffusion_batch_size", "diffusion_chunk_size"])
def test_explicit_diffusion_config_rejects_non_protenix_model(key: str) -> None:
    flag = "--" + key.replace("_", "-")

    with pytest.raises(SystemExit) as error:
        train.parse_args(["train.cif", "--model", "proteinmpnn", flag, "2"])

    assert error.value.code == 2


@pytest.mark.parametrize("key", ["diffusion_batch_size", "diffusion_chunk_size"])
def test_runner_config_rejects_known_fresh_non_protenix_model(key: str) -> None:
    with pytest.raises(ValueError, match="Protenix|protenix"):
        RunnerConfig({"model": {"name": "proteinmpnn"}, key: 2})


def test_runner_wires_diffusion_controls_into_actual_collate_without_polluting_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_danling_init(self: Any, config: RunnerConfig) -> None:
        self.config = config
        self.datasets = {}

    def criterion_factory(model: Any, processor: Any) -> Any:
        del model, processor
        return lambda prediction, target: (prediction, target)

    monkeypatch.setattr(runner_module.dl.Runner, "__init__", fake_danling_init)
    # DanLing's metaclass invokes ``__post_init__`` after the real OFoldX
    # ``Runner.__init__`` returns. The fake base initializer intentionally does
    # not construct DanLing fault-tolerance state, so isolate that unrelated
    # lifecycle hook while retaining the Runner wiring under test.
    monkeypatch.setattr(Runner, "__post_init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture: (criterion_factory, _sentinel_collate, True, True),
    )
    monkeypatch.setattr(
        Runner,
        "_resolve_registration",
        lambda self, config, repository: SimpleNamespace(id="protenix"),
    )
    monkeypatch.setattr(Runner, "_build", lambda self, repository, registration: (_FakeModel(), object()))
    monkeypatch.setattr(Runner, "build_datasets", lambda self: {})

    runner = Runner(
        RunnerConfig(
            {
                "stack": "ddp",
                "model": {"name": "protenix"},
                "objective_recipe": "ofoldx_edm_v1",
                "data": {},
                "logging": {"enabled": False},
                "diffusion_batch_size": 5,
                "diffusion_chunk_size": 2,
            }
        )
    )
    output = runner.collate_fn(["item"])

    assert output["input"] == {
        "input": "features",
        "target": "truth",
        "num_samples": 5,
        "sample_chunk_size": 2,
        "compute_auxiliary_heads": False,
    }
    assert output["target"] == {"input": "features", "target": "truth"}
    prediction, criterion_target = runner.criterion("prediction", output["target"])
    assert prediction == "prediction"
    assert criterion_target is output["target"]
    assert "num_samples" not in output["target"]
    assert "sample_chunk_size" not in output["target"]
    assert "compute_auxiliary_heads" not in output["target"]
    assert runner.model.training is True


def test_runner_rejects_diffusion_control_after_pretrained_architecture_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Runner, "_prepare_repository", lambda self, config: object())
    monkeypatch.setattr(
        Runner,
        "_resolve_registration",
        lambda self, config, repository: SimpleNamespace(id="proteinmpnn"),
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture: (object(), _sentinel_collate, False, False),
    )
    config = RunnerConfig(
        {
            "stack": "ddp",
            "model": {"pretrained": "organization/model"},
            "data": {},
            "diffusion_batch_size": 2,
        }
    )

    with pytest.raises(ValueError, match="Protenix|protenix"):
        Runner(config)


def test_collate_adds_only_controlled_model_kwargs() -> None:
    output = collate_with_model_kwargs(
        ["item"],
        collate_fn=_sentinel_collate,
        model_kwargs={"num_samples": 5, "sample_chunk_size": 2, "compute_auxiliary_heads": False},
    )

    assert output["input"] == {
        "input": "features",
        "target": "truth",
        "num_samples": 5,
        "sample_chunk_size": 2,
        "compute_auxiliary_heads": False,
    }
    assert output["target"] == {"input": "features", "target": "truth"}
    assert output["loss_normalizer"] == 7


@pytest.mark.parametrize("reserved", ["input", "target"])
def test_collate_model_kwargs_cannot_overwrite_inputs(reserved: str) -> None:
    with pytest.raises(ValueError, match=reserved):
        collate_with_model_kwargs(
            ["item"],
            collate_fn=_sentinel_collate,
            model_kwargs={reserved: "replacement"},
        )


def test_top_level_collate_partial_is_picklable() -> None:
    collate = partial(
        collate_with_model_kwargs,
        collate_fn=_sentinel_collate,
        model_kwargs={"num_samples": 5, "sample_chunk_size": 2},
    )

    assert pickle.dumps(collate)
