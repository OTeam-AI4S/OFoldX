from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import ofoldx.runners.train as canonical
from ofoldx.runners import EMAShadowModel, Runner, RunnerConfig


class _RichModel(nn.Module):
    def __init__(
        self,
        *,
        width: int = 2,
        real_dtype: torch.dtype = torch.float32,
        extra_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, width + 1, dtype=real_dtype))
        self.complex_weight = nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=torch.complex64))
        if extra_parameter:
            self.extra = nn.Parameter(torch.ones(1, dtype=real_dtype))
        self.register_buffer("float_buffer", torch.tensor([1.5], dtype=real_dtype))
        self.register_buffer("bool_buffer", torch.tensor([True, False]))
        self.register_buffer("int_buffer", torch.tensor([3, 4], dtype=torch.int64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight.sum()

    def sample(self, value: torch.Tensor, *, offset: float = 0.0) -> torch.Tensor:
        return self(value) + offset


def _assert_same_state(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for name in left_state:
        assert torch.equal(left_state[name], right_state[name]), name


@pytest.mark.parametrize("decay", [0, 1, -0.1, float("nan"), float("inf")])
def test_ema_shadow_rejects_decay_outside_the_open_unit_interval(decay: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        EMAShadowModel(_RichModel(), decay)


@pytest.mark.parametrize("decay", [True, "0.9", None])
def test_ema_shadow_rejects_non_numeric_decay(decay: Any) -> None:
    with pytest.raises(TypeError, match="decay"):
        EMAShadowModel(_RichModel(), decay)


def test_ema_shadow_starts_as_an_independent_frozen_copy() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.9)

    _assert_same_state(shadow, source)
    assert list(shadow.state_dict()) == list(source.state_dict())
    for source_parameter, shadow_parameter in zip(source.parameters(), shadow.parameters()):
        assert shadow_parameter.requires_grad is False
        assert source_parameter.data_ptr() != shadow_parameter.data_ptr()

    with torch.no_grad():
        source.weight.add_(100)
        source.float_buffer.add_(100)
    assert not torch.equal(shadow.state_dict()["weight"], source.state_dict()["weight"])
    assert not torch.equal(shadow.state_dict()["float_buffer"], source.state_dict()["float_buffer"])


def test_ema_update_applies_fixed_decay_repeatedly_to_real_and_complex_parameters() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.75)
    initial_real = source.weight.detach().clone()
    initial_complex = source.complex_weight.detach().clone()

    with torch.no_grad():
        source.weight.fill_(5)
        source.complex_weight.fill_(5 + 6j)
    shadow.update(source)
    expected_real = initial_real * 0.75 + source.weight.detach() * 0.25
    expected_complex = initial_complex * 0.75 + source.complex_weight.detach() * 0.25
    assert torch.equal(shadow.state_dict()["weight"], expected_real)
    assert torch.equal(shadow.state_dict()["complex_weight"], expected_complex)

    previous_real = expected_real.clone()
    previous_complex = expected_complex.clone()
    with torch.no_grad():
        source.weight.fill_(-3)
        source.complex_weight.fill_(-2 + 4j)
    shadow.update(source)
    assert torch.equal(
        shadow.state_dict()["weight"],
        previous_real * 0.75 + source.weight.detach() * 0.25,
    )
    assert torch.equal(
        shadow.state_dict()["complex_weight"],
        previous_complex * 0.75 + source.complex_weight.detach() * 0.25,
    )


def test_ema_update_copies_every_buffer_exactly_instead_of_averaging() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.999)

    with torch.no_grad():
        source.float_buffer.fill_(9.5)
        source.bool_buffer.copy_(torch.tensor([False, True]))
        source.int_buffer.copy_(torch.tensor([-7, 11]))
    shadow.update(source)

    for name in ("float_buffer", "bool_buffer", "int_buffer"):
        assert torch.equal(shadow.state_dict()[name], source.state_dict()[name])


def test_ema_copy_from_synchronizes_parameters_and_buffers_without_averaging() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.999)

    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(7)
        source.float_buffer.fill_(8)
        source.bool_buffer.zero_()
        source.int_buffer.fill_(9)
    shadow.copy_from(source)

    _assert_same_state(shadow, source)


@pytest.mark.parametrize(
    ("incompatible", "message"),
    [
        (_RichModel(extra_parameter=True), "names do not match"),
        (_RichModel(width=3), "shape mismatch"),
        (_RichModel(real_dtype=torch.float64), "dtype mismatch"),
    ],
    ids=["keys", "shape", "dtype"],
)
@pytest.mark.parametrize("operation", ["update", "copy_from"])
def test_ema_rejects_incompatible_source_before_mutating(
    incompatible: nn.Module,
    message: str,
    operation: str,
) -> None:
    shadow = EMAShadowModel(_RichModel(), decay=0.9)
    before = {name: value.clone() for name, value in shadow.state_dict().items()}

    with pytest.raises(ValueError, match=message):
        getattr(shadow, operation)(incompatible)

    for name, value in shadow.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_ema_delegates_forward_and_sample_to_the_shadow_copy() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.9)
    value = torch.tensor([2.0])

    assert torch.equal(shadow(value), source(value))
    assert torch.equal(shadow.sample(value, offset=3.0), source.sample(value, offset=3.0))


def test_ema_state_dict_round_trips_with_ordinary_model_keys() -> None:
    source = _RichModel()
    shadow = EMAShadowModel(source, decay=0.9)
    with torch.no_grad():
        source.weight.fill_(12)
        source.complex_weight.fill_(3 - 4j)
        source.int_buffer.fill_(13)
    shadow.update(source)
    state = {name: value.clone() for name, value in shadow.state_dict().items()}

    restored = EMAShadowModel(_RichModel(), decay=0.5)
    restored.load_state_dict(state)

    assert restored.decay == 0.5
    assert list(state) == list(source.state_dict())
    assert not any(name.startswith("model.") for name in state)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, state[name]), name
    assert all(not parameter.requires_grad for parameter in restored.parameters())


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.backend = "auto"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight

    def sample(self, value: torch.Tensor) -> torch.Tensor:
        return self(value)

    def set_backend(self, backend: str) -> _ScalarModel:
        self.backend = backend
        return self


class _CompileSchemaWrapper(nn.Module):
    r"""Minimal compile-like wrapper whose checkpoint keys gain ``_orig_mod.``."""

    def __init__(self, model: _ScalarModel) -> None:
        super().__init__()
        self._orig_mod = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(value)


class _Processor:
    config: dict[str, Any] = {}


class _TinyRunner(Runner):
    def _build(self, repository: Any, registration: Any) -> tuple[nn.Module, _Processor]:
        del repository, registration
        return _ScalarModel(), _Processor()


class _MaterializingTinyRunner(_TinyRunner):
    def materialize_model(self) -> None:
        # Simulate a materialization-time source change after Runner created
        # the shadow but before Runner's synchronization hook runs.
        with torch.no_grad():
            self.model.weight.fill_(7)
        super().materialize_model()


def _live_scalar_model(runner: Runner) -> _ScalarModel:
    model = runner.unwrap(runner.model)
    model = getattr(model, "_orig_mod", model)
    assert isinstance(model, _ScalarModel)
    return model


def _tiny_config(tmp_path: Path, *, ema_decay: float | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": {"name": "proteinmpnn"},
        "epochs": 1,
        "optim": {"type": "torch_adamw", "lr": 0.1, "weight_decay": 0.0},
        "logging": {"enabled": False},
        "workspace": {"root": str(tmp_path / "workspace")},
        "ckpt": {"enabled": False},
    }
    if ema_decay is not None:
        config["ema_decay"] = ema_decay
    return config


@pytest.fixture
def tiny_objective(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def resolve_objective(architecture: str) -> tuple[Any, Any, bool, bool]:
        assert architecture == "proteinmpnn"
        return (lambda model, processor: nn.MSELoss(), lambda items: items, False, False)

    monkeypatch.setattr("ofoldx.runners.runner._resolve_objective", resolve_objective)
    yield


def test_runner_leaves_ema_unbound_and_config_key_absent_by_default(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path))
    try:
        assert runner.ema is None
        assert "ema_decay" not in runner.config
    finally:
        runner.close()


def test_runner_binds_frozen_ema_and_excludes_it_from_the_optimizer(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.8))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        assert runner.ema.decay == 0.8
        assert all(not parameter.requires_grad for parameter in runner.ema.parameters())
        optimizer_parameters = {
            id(parameter)
            for group in runner.optimizer.param_groups
            for parameter in group["params"]
        }
        assert id(_live_scalar_model(runner).weight) in optimizer_parameters
        assert optimizer_parameters.isdisjoint(id(parameter) for parameter in runner.ema.parameters())
    finally:
        runner.close()


def test_danling_optimizer_step_updates_runner_ema_automatically(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.5))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        source_model = _live_scalar_model(runner)
        source_before = source_model.weight.detach().clone()
        shadow_before = runner.ema.state_dict()["weight"].clone()
        source_model.weight.grad = torch.ones_like(source_model.weight)

        assert runner.optimizer_step() is True

        source_after = source_model.weight.detach()
        shadow_after = runner.ema.state_dict()["weight"]
        assert not torch.equal(source_after, source_before)
        assert torch.allclose(shadow_after, shadow_before * 0.5 + source_after * 0.5)
        assert runner.train_state.global_step == 1
    finally:
        runner.close()


def test_runner_resynchronizes_ema_after_model_materialization(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _MaterializingTinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        assert _live_scalar_model(runner).weight.item() == 7
        assert runner.ema.state_dict()["weight"].item() == 7
    finally:
        runner.close()


@pytest.mark.parametrize("payload_key", ["model", "ema"])
def test_runner_loads_unprefixed_checkpoint_through_compile_schema_and_resynchronizes_ema(
    tmp_path: Path,
    tiny_objective: None,
    payload_key: str,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        wrapper = _CompileSchemaWrapper(_live_scalar_model(runner))
        assert list(wrapper.state_dict()) == ["_orig_mod.weight"]
        with pytest.raises(RuntimeError, match="Missing key.*_orig_mod.weight"):
            wrapper.load_state_dict({"weight": torch.tensor(3.0)})
        runner.model = wrapper

        runner.load_model({"weight": torch.tensor(5.0)})
        assert _live_scalar_model(runner).weight.item() == 5

        runner.load_pretrained({payload_key: {"weight": torch.tensor(11.0)}})
        assert _live_scalar_model(runner).weight.item() == 11
        assert runner.ema.state_dict()["weight"].item() == 11
    finally:
        runner.close()


def test_full_checkpoint_from_compile_schema_restores_live_and_ema_separately(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        runner.model = _CompileSchemaWrapper(_live_scalar_model(runner))
        with torch.no_grad():
            _live_scalar_model(runner).weight.fill_(3)
            runner.ema.model.weight.fill_(7)

        checkpoint = dict(runner.state_dict())
        assert list(checkpoint["model"]) == ["_orig_mod.weight"]
        assert list(checkpoint["ema"]) == ["weight"]
        checkpoint["model"] = {name: value.clone() for name, value in checkpoint["model"].items()}
        checkpoint["ema"] = {name: value.clone() for name, value in checkpoint["ema"].items()}
        assert checkpoint["model"]["_orig_mod.weight"].item() == 3
        assert checkpoint["ema"]["weight"].item() == 7

        with torch.no_grad():
            _live_scalar_model(runner).weight.fill_(13)
            runner.ema.model.weight.fill_(17)
        runner.load_checkpoint(checkpoint)

        assert _live_scalar_model(runner).weight.item() == 3
        assert runner.ema.state_dict()["weight"].item() == 7
    finally:
        runner.close()


@pytest.mark.parametrize("compile_target", [False, True], ids=["raw-target", "compile-target"])
def test_legacy_compile_prefixed_pretrained_payload_loads_into_current_model_schema(
    tmp_path: Path,
    tiny_objective: None,
    compile_target: bool,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        if compile_target:
            runner.model = _CompileSchemaWrapper(_live_scalar_model(runner))

        runner.load_pretrained({"model": {"_orig_mod.weight": torch.tensor(19.0)}})

        assert _live_scalar_model(runner).weight.item() == 19
        assert runner.ema.state_dict()["weight"].item() == 19
    finally:
        runner.close()


def test_danling_evaluate_step_prefers_ema_over_live_model(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        assert isinstance(runner.ema, EMAShadowModel)
        with torch.no_grad():
            _live_scalar_model(runner).weight.fill_(2)
            runner.ema.model.weight.fill_(5)

        prediction, loss = runner.evaluate_step(
            {"input": torch.tensor(3.0), "target": torch.tensor(0.0)}
        )

        assert prediction.item() == 15
        assert loss is not None and loss.item() == 225
    finally:
        runner.close()


def test_full_state_resume_requires_ema_when_the_runner_has_ema_enabled(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    runner = _TinyRunner(_tiny_config(tmp_path, ema_decay=0.9))
    try:
        checkpoint = dict(runner.state_dict())
        checkpoint.pop("ema")

        with pytest.raises(ValueError, match="checkpoint has no EMA state"):
            runner.load_checkpoint(checkpoint)
    finally:
        runner.close()


def test_ema_cli_is_disabled_without_changing_legacy_config_identity() -> None:
    args = canonical.parse_args(["train.cif"])
    config = canonical.build_config(args)

    assert args.ema_decay is None
    assert "ema_decay" not in config


def test_ema_cli_maps_explicit_decay_to_top_level_config() -> None:
    args = canonical.parse_args(["train.cif", "--ema-decay", "0.999"])
    config = canonical.build_config(args)

    assert args.ema_decay == pytest.approx(0.999)
    assert config["ema_decay"] == pytest.approx(0.999)


@pytest.mark.parametrize("value", ["0", "1", "-0.1", "nan", "inf", "-inf", "true"])
def test_ema_cli_rejects_invalid_decay(value: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["train.cif", "--ema-decay", value])


def test_ema_yaml_matches_equivalent_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "ema.yaml"
    config_path.write_text("train_source: train.cif\nema_decay: 0.875\n", encoding="utf-8")

    yaml_args = canonical.parse_args(["--config", str(config_path)])
    cli_args = canonical.parse_args(["train.cif", "--ema-decay", "0.875"])

    assert yaml_args.ema_decay == pytest.approx(0.875)
    assert canonical.build_config(yaml_args) == canonical.build_config(cli_args)


@pytest.mark.parametrize("ema_line", ["", "ema_decay: null\n"], ids=["omitted", "null"])
def test_ema_yaml_disabled_state_omits_the_config_key(tmp_path: Path, ema_line: str) -> None:
    config_path = tmp_path / "ema-disabled.yaml"
    config_path.write_text(f"train_source: train.cif\n{ema_line}", encoding="utf-8")

    args = canonical.parse_args(["--config", str(config_path)])
    config = canonical.build_config(args)

    assert args.ema_decay is None
    assert "ema_decay" not in config


@pytest.mark.parametrize(
    "value",
    ["0", "1", "-0.1", ".nan", ".inf", "true", '"0.9"'],
    ids=["zero", "one", "negative", "nan", "infinity", "boolean", "string"],
)
def test_ema_yaml_rejects_invalid_decay(tmp_path: Path, value: str) -> None:
    config_path = tmp_path / "invalid-ema.yaml"
    config_path.write_text(f"train_source: train.cif\nema_decay: {value}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        canonical.parse_args(["--config", str(config_path)])


@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan"), float("inf"), True, "0.9"])
def test_programmatic_runner_config_rejects_invalid_ema_decay(value: Any) -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        RunnerConfig({"model": {"name": "proteinmpnn"}, "ema_decay": value})


def test_programmatic_runner_config_normalizes_ema_and_drops_explicit_none() -> None:
    enabled = RunnerConfig({"model": {"name": "proteinmpnn"}, "ema_decay": 0.75})
    disabled = RunnerConfig({"model": {"name": "proteinmpnn"}, "ema_decay": None})

    assert enabled.ema_decay == pytest.approx(0.75)
    assert "ema_decay" not in disabled
