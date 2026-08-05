from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from danling.metrics import AverageMeters
from torch import nn

import ofoldx.runners.runner as runner_module
import ofoldx.runners.train as train
from ofoldx.models.protenix.model import (
    ProtenixEvaluationOutput,
    ProtenixRawOutput,
    protenix_contact_probability,
)
from ofoldx.models.protenix.permutation import ProtenixPredictionPermutation
from ofoldx.runners.collate import collate_protenix_baseline
from ofoldx.runners.config import OFOLDX_EDM_RECIPE, PROTENIX_BASELINE_RECIPE, RunnerConfig
from ofoldx.runners.runner import Runner


class _FakeModel:
    def __init__(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> _FakeModel:
        self.training = mode
        return self


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return value * self.weight


class _Processor:
    config: dict[str, Any] = {}

    def process_training(self, value: Any) -> Any:
        return value


class _TinyProtenixRunner(Runner):
    def _build(self, repository: Any, registration: Any) -> tuple[nn.Module, _Processor]:
        del repository, registration
        return _ScalarModel(), _Processor()


class _FiniteCriterion:
    def __init__(self) -> None:
        self.latest_metrics: dict[str, torch.Tensor] = {}

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (prediction - target).square().mean()
        self.latest_metrics = {
            "objective/mse": loss.detach(),
            "objective/weighted_mse": loss.detach() * 4,
        }
        return loss


class _SamplingModel(nn.Module):
    def __init__(self, prediction: ProtenixRawOutput) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.prediction = prediction
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def sample(self, model_input: Any, **kwargs: Any) -> ProtenixRawOutput:
        self.calls.append((model_input, kwargs))
        return self.prediction


class _EvaluationCriterion:
    def __init__(self, loss: float = 3.0) -> None:
        self.loss = loss
        self.latest_metrics: dict[str, torch.Tensor] = {}
        self.calls: list[tuple[ProtenixRawOutput, dict[str, Any]]] = []

    def evaluate(self, prediction: ProtenixRawOutput, target: dict[str, Any]) -> torch.Tensor:
        self.calls.append((prediction, target))
        self.latest_metrics = {
            "objective/mse": torch.tensor(2.0),
            "objective/weighted_mse": torch.tensor(8.0),
            "objective/loss": torch.tensor(self.loss),
        }
        return torch.tensor(self.loss)


def _raw_evaluation_prediction(*, atom_count: int = 3, token_count: int = 2) -> ProtenixRawOutput:
    sample_count = 2
    pair_id = torch.arange(token_count * token_count, dtype=torch.float32).reshape(
        1, token_count, token_count
    )
    return ProtenixRawOutput(  # type: ignore[call-arg]
        coordinates=torch.arange(1 * sample_count * atom_count * 3, dtype=torch.float32).reshape(
            1, sample_count, atom_count, 3
        ),
        distogram_logits=torch.stack((pair_id, torch.zeros_like(pair_id)), dim=-1),
        plddt_logits=torch.arange(1 * sample_count * atom_count * 2, dtype=torch.float32).reshape(
            1, sample_count, atom_count, 2
        ),
        pae_logits=torch.arange(
            1 * sample_count * token_count * token_count * 2, dtype=torch.float32
        ).reshape(1, sample_count, token_count, token_count, 2),
        pde_logits=torch.arange(
            1 * sample_count * token_count * token_count * 2, dtype=torch.float32
        ).reshape(1, sample_count, token_count, token_count, 2)
        + 100,
        resolved_logits=torch.arange(1 * sample_count * atom_count * 2, dtype=torch.float32).reshape(
            1, sample_count, atom_count, 2
        )
        + 200,
        batch_size=[1],
    )


def _mock_baseline_evaluation_runner(
    model: nn.Module,
    criterion: _EvaluationCriterion,
    *,
    ema: nn.Module | None,
) -> Runner:
    runner = object.__new__(Runner)
    runner._objective_recipe = PROTENIX_BASELINE_RECIPE
    runner.model = model
    runner.ema = ema
    runner.criterion = criterion
    runner.metrics = None
    runner.meters = AverageMeters()
    for name in runner_module._PROTENIX_BASELINE_METRIC_KEYS:
        runner.meters[name].update(0.0, n=0)
    runner._baseline_current_metric_keys = frozenset()
    runner.to_device = lambda data: data
    runner.infer_context = nullcontext
    runner._draw_baseline_mc_dropout = lambda: True
    runner.train_state = SimpleNamespace(global_step=11, micro_step=7)
    runner.backward = lambda loss: pytest.fail("evaluation must not run backward")
    runner.step = lambda: pytest.fail("evaluation must not advance the optimizer")
    return runner


def _sentinel_collate(batch: list[Any]) -> dict[str, Any]:
    assert batch == ["item"]
    pair = {"input": "features", "target": "truth"}
    return {"input": pair, "target": pair, "loss_normalizer": 1}


def _tiny_config(workspace: Path, recipe: str | None, *, model_name: str = "protenix") -> dict[str, Any]:
    config: dict[str, Any] = {
        "stack": "ddp",
        "model": {"name": model_name},
        "epochs": 1,
        "optim": {"type": "sgd", "lr": 0.1, "weight_decay": 0.1},
        "logging": {"enabled": False},
        "workspace": {"dir": str(workspace)},
        "ckpt": {"enabled": False},
    }
    if recipe is not None:
        config["objective_recipe"] = recipe
    return config


@pytest.fixture
def tiny_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    def criterion_factory(model: Any, processor: Any, objective_config: Any = None) -> _FiniteCriterion:
        del model, processor, objective_config
        return _FiniteCriterion()

    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture, recipe=OFOLDX_EDM_RECIPE: (
            criterion_factory,
            _sentinel_collate,
            True,
            True,
        ),
    )


def test_fresh_recipe_defaults_are_materialized_before_runner_identity() -> None:
    baseline = RunnerConfig({"model": {"name": "protenix"}})
    baseline.boot()
    edm = RunnerConfig({"model": {"name": "boltz1"}})
    edm.boot()

    assert baseline.objective_recipe == PROTENIX_BASELINE_RECIPE
    assert baseline.diffusion_batch_size == 48
    assert baseline.diffusion_chunk_size == 4
    assert "objective_recipe" not in edm
    assert edm.materialize_objective_recipe("boltz1") == OFOLDX_EDM_RECIPE
    assert "diffusion_batch_size" not in edm
    assert "diffusion_chunk_size" not in edm
    assert hash(baseline) != hash(
        RunnerConfig({"model": {"name": "protenix"}, "objective_recipe": OFOLDX_EDM_RECIPE})
    )


def test_non_protenix_absent_and_explicit_edm_have_the_same_mapping_and_hash() -> None:
    implicit = RunnerConfig({"model": {"name": "proteinmpnn"}})
    explicit = RunnerConfig({"model": {"name": "proteinmpnn"}, "objective_recipe": OFOLDX_EDM_RECIPE})
    implicit.boot()
    explicit.boot()

    assert "objective_recipe" not in implicit
    assert "objective_recipe" not in explicit
    assert dict(implicit) == dict(explicit)
    assert hash(implicit) == hash(explicit)


def test_explicit_protenix_edm_remains_persisted_without_baseline_defaults() -> None:
    config = RunnerConfig({"model": {"name": "protenix"}, "objective_recipe": OFOLDX_EDM_RECIPE})
    config.boot()

    assert config.objective_recipe == OFOLDX_EDM_RECIPE
    assert "objective_recipe" in config
    assert "diffusion_batch_size" not in config
    assert "diffusion_chunk_size" not in config


def test_unresolved_pretrained_recipe_defers_until_registration_is_known() -> None:
    explicit_edm = RunnerConfig(
        {"model": {"pretrained": "organization/model"}, "objective_recipe": OFOLDX_EDM_RECIPE}
    )
    implicit = RunnerConfig({"model": {"pretrained": "organization/model"}})

    assert explicit_edm.materialize_objective_recipe() is None
    assert explicit_edm.objective_recipe == OFOLDX_EDM_RECIPE
    assert implicit.materialize_objective_recipe() is None
    assert "objective_recipe" not in implicit

    assert explicit_edm.materialize_objective_recipe("proteinmpnn") == OFOLDX_EDM_RECIPE
    assert "objective_recipe" not in explicit_edm
    assert implicit.materialize_objective_recipe("protenix") == PROTENIX_BASELINE_RECIPE
    assert implicit.objective_recipe == PROTENIX_BASELINE_RECIPE
    assert implicit.diffusion_batch_size == 48
    assert implicit.diffusion_chunk_size == 4


def test_unresolved_pretrained_baseline_is_rejected_after_non_protenix_resolution() -> None:
    config = RunnerConfig(
        {
            "model": {"pretrained": "organization/model"},
            "objective_recipe": PROTENIX_BASELINE_RECIPE,
        }
    )

    assert config.materialize_objective_recipe() is None
    with pytest.raises(ValueError, match="only by Protenix"):
        config.materialize_objective_recipe("proteinmpnn")


@pytest.mark.parametrize("recipe", [None, "unknown_v1", 1, True])
def test_invalid_objective_recipe_is_rejected(recipe: Any) -> None:
    with pytest.raises(ValueError, match="objective_recipe"):
        RunnerConfig({"model": {"name": "protenix"}, "objective_recipe": recipe})


def test_non_protenix_rejects_baseline_recipe() -> None:
    with pytest.raises(ValueError, match="only by Protenix"):
        RunnerConfig(
            {"model": {"name": "proteinmpnn"}, "objective_recipe": PROTENIX_BASELINE_RECIPE}
        )


def test_baseline_collate_is_structure_normalized_without_mutating_pair() -> None:
    model_input = object()
    model_target = object()

    collated = collate_protenix_baseline([(model_input, model_target)])

    assert collated["input"] == {"input": model_input, "target": model_target}
    assert collated["target"] is collated["input"]
    assert collated["loss_normalizer"] == 1


@pytest.mark.parametrize("configured_recipe", [None, OFOLDX_EDM_RECIPE])
def test_runner_resolves_non_protenix_to_internal_edm_without_persisting_recipe(
    monkeypatch: pytest.MonkeyPatch,
    configured_recipe: str | None,
) -> None:
    observed: dict[str, Any] = {}

    def fake_danling_init(self: Any, config: RunnerConfig) -> None:
        self.config = config
        self.datasets = {}
        self.meters = AverageMeters()

    def resolve_objective(architecture: str, recipe: str = OFOLDX_EDM_RECIPE) -> Any:
        observed["architecture"] = architecture
        observed["recipe"] = recipe
        return (
            lambda model, processor: _FiniteCriterion(),
            _sentinel_collate,
            False,
            False,
        )

    monkeypatch.setattr(runner_module.dl.Runner, "__init__", fake_danling_init)
    monkeypatch.setattr(Runner, "__post_init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(runner_module, "_resolve_objective", resolve_objective)
    monkeypatch.setattr(
        Runner,
        "_resolve_registration",
        lambda self, config, repository: SimpleNamespace(id="proteinmpnn"),
    )
    monkeypatch.setattr(Runner, "_build", lambda self, repository, registration: (_FakeModel(), object()))
    monkeypatch.setattr(Runner, "build_datasets", lambda self: {})
    values: dict[str, Any] = {
        "stack": "ddp",
        "model": {"name": "proteinmpnn"},
        "data": {},
        "logging": {"enabled": False},
    }
    if configured_recipe is not None:
        values["objective_recipe"] = configured_recipe

    runner = Runner(RunnerConfig(values))

    assert observed == {"architecture": "proteinmpnn", "recipe": OFOLDX_EDM_RECIPE}
    assert runner._objective_recipe == OFOLDX_EDM_RECIPE
    assert runner._resolved_architecture == "proteinmpnn"
    assert "objective_recipe" not in runner.config


def test_runner_wires_baseline_forward_recipe_samples_chunks_and_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_danling_init(self: Any, config: RunnerConfig) -> None:
        self.config = config
        self.datasets = {}
        self.meters = AverageMeters()

    def criterion_factory(model: Any, processor: Any, objective_config: Any = None) -> Any:
        del model, processor
        assert objective_config is not None
        return lambda prediction, target: (prediction, target)

    monkeypatch.setattr(runner_module.dl.Runner, "__init__", fake_danling_init)
    monkeypatch.setattr(Runner, "__post_init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture, recipe=OFOLDX_EDM_RECIPE: (
            criterion_factory,
            _sentinel_collate,
            True,
            True,
        ),
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
                "data": {},
                "logging": {"enabled": False},
            }
        )
    )
    output = runner.collate_fn(["item"])

    assert output["input"] == {
        "input": "features",
        "target": "truth",
        "training_recipe": PROTENIX_BASELINE_RECIPE,
        "compute_auxiliary_heads": True,
        "mini_rollout_steps": 20,
        "train_confidence_only": False,
        "num_samples": 48,
        "sample_chunk_size": 4,
    }
    assert output["target"] == {"input": "features", "target": "truth"}
    assert runner._objective_recipe == PROTENIX_BASELINE_RECIPE
    assert runner.config.objective_config.alpha_diffusion == 4.0
    assert runner.config.objective_config.compute_zero_weight_terms is True
    assert runner.model.training is True


def test_runner_wires_validated_confidence_only_stage_to_model_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_danling_init(self: Any, config: RunnerConfig) -> None:
        self.config = config
        self.datasets = {}
        self.meters = AverageMeters()

    monkeypatch.setattr(runner_module.dl.Runner, "__init__", fake_danling_init)
    monkeypatch.setattr(Runner, "__post_init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture, recipe=OFOLDX_EDM_RECIPE: (
            lambda model, processor, objective_config=None: _FiniteCriterion(),
            _sentinel_collate,
            True,
            True,
        ),
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
                "objective_config": {"train_confidence_only": True},
                "data": {},
                "logging": {"enabled": False},
            }
        )
    )
    output = runner.collate_fn(["item"])

    assert output["input"]["train_confidence_only"] is True
    assert runner.config.objective_config.train_confidence_only is True
    assert runner.config.objective_config.alpha_diffusion == 4.0
    assert runner.config.objective_config.alpha_distogram == pytest.approx(0.03)


def test_baseline_train_step_injects_global_step_and_records_finite_metrics() -> None:
    observed: dict[str, Any] = {"steps": 0}

    class ObservedModel(_ScalarModel):
        def forward(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            observed["model_kwargs"] = kwargs
            return super().forward(value, **kwargs)

    runner = object.__new__(Runner)
    runner._objective_recipe = PROTENIX_BASELINE_RECIPE
    runner.train_state = SimpleNamespace(global_step=17)
    runner.model = ObservedModel()
    runner.criterion = _FiniteCriterion()
    runner.metrics = None
    runner.meters = AverageMeters()
    for name in runner_module._PROTENIX_BASELINE_METRIC_KEYS:
        runner.meters[name].update(0.0, n=0)
    runner._baseline_current_metric_keys = frozenset()
    runner.to_device = lambda data: data
    runner.train_context = nullcontext
    runner.backward = lambda loss: loss.backward()

    def step() -> None:
        observed["steps"] += 1

    runner.step = step
    original = {"input": {"value": torch.tensor(2.0)}, "target": torch.tensor(0.0)}

    prediction, loss = runner.train_step(original)

    assert torch.equal(prediction, torch.tensor(2.0))
    assert loss is not None and loss.item() == 4.0
    assert observed["model_kwargs"]["current_step"] == 17
    assert "current_step" not in original["input"]
    assert observed["steps"] == 1
    assert runner.meters["objective/mse"].count == 1
    assert runner.get_step_result()["objective/mse"] == 4.0


def test_nonfinite_baseline_loss_restores_gradients_and_hides_stale_metrics() -> None:
    class NonfiniteCriterion:
        latest_metrics = {"objective/mse": torch.tensor(99.0)}

        def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            del target
            return prediction.sum() * torch.tensor(float("nan"))

    runner = object.__new__(Runner)
    runner._objective_recipe = PROTENIX_BASELINE_RECIPE
    runner.train_state = SimpleNamespace(global_step=3)
    runner.model = _ScalarModel()
    runner.model.weight.grad = torch.tensor(7.0)
    runner.criterion = NonfiniteCriterion()
    runner.metrics = None
    runner.meters = AverageMeters()
    for name in runner_module._PROTENIX_BASELINE_METRIC_KEYS:
        runner.meters[name].update(0.0, n=0)
    runner.meters["objective/mse"].update(5.0)
    runner._baseline_current_metric_keys = frozenset({"objective/mse"})
    runner.to_device = lambda data: data
    runner.train_context = nullcontext
    runner.backward = lambda loss: loss.backward()
    runner.step = lambda: None
    runner.iter_optimizer_parameters = lambda: iter(runner.model.parameters())
    runner._train_no_sync_targets = lambda: ()
    runner._should_train_no_sync = lambda: False

    _, loss = runner.train_step(
        {"input": {"value": torch.tensor(2.0)}, "target": torch.tensor(0.0)}
    )

    assert loss is not None and loss.item() == 0.0
    assert runner.model.weight.grad is not None
    assert runner.model.weight.grad.item() == 7.0
    assert runner.meters["objective/mse"].count == 1
    assert "objective/mse" not in runner.get_step_result()


@pytest.mark.parametrize(("draw", "expected"), [(0.399, True), (0.4, False)])
def test_baseline_mc_dropout_uses_python_rng_contract(
    monkeypatch: pytest.MonkeyPatch,
    draw: float,
    expected: bool,
) -> None:
    torch_state = torch.random.get_rng_state()
    monkeypatch.setattr(runner_module.random, "random", lambda: draw)

    assert Runner._draw_baseline_mc_dropout() is expected
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_baseline_evaluate_step_uses_ema_defaults_permutation_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_prediction = _raw_evaluation_prediction()
    live_model = _SamplingModel(raw_prediction).train()
    ema_model = _SamplingModel(raw_prediction).train()
    criterion = _EvaluationCriterion()
    runner = _mock_baseline_evaluation_runner(live_model, criterion, ema=ema_model)
    observed_generic_metrics: list[tuple[Any, Any]] = []
    runner.metrics = SimpleNamespace(
        update=lambda prediction, target: observed_generic_metrics.append((prediction, target)),
        value=lambda: {},
    )
    model_input = SimpleNamespace(tfg_guidance=object())
    structure_target = object()
    atom_permutation = torch.tensor([[[2, 0, 1], [1, 2, 0]]])
    token_permutation = torch.tensor([[[1, 0], [0, 1]]])
    permutation = ProtenixPredictionPermutation(
        coordinates=raw_prediction.coordinates.gather(
            2,
            atom_permutation.unsqueeze(-1).expand_as(raw_prediction.coordinates),
        ),
        atom_permutation=atom_permutation,
        token_permutation=token_permutation,
    )

    def match_prediction(coordinates: torch.Tensor, target: Any, input: Any) -> Any:
        assert coordinates is raw_prediction.coordinates
        assert target is structure_target
        assert input is model_input
        return permutation

    monkeypatch.setattr(
        "ofoldx.models.protenix.permutation.match_protenix_prediction",
        match_prediction,
    )
    live_weight = live_model.weight.detach().clone()
    ema_weight = ema_model.weight.detach().clone()
    train_state_before = dict(vars(runner.train_state))
    data = {
        "input": {
            "input": model_input,
            "target": structure_target,
            "training_recipe": PROTENIX_BASELINE_RECIPE,
            "compute_auxiliary_heads": True,
            "num_samples": 48,
            "sample_chunk_size": 4,
        },
        "target": {"input": model_input, "target": structure_target},
    }

    prediction, loss = runner.evaluate_step(data)

    assert not ema_model.training
    assert live_model.training
    assert not live_model.calls
    assert len(ema_model.calls) == 1
    sampled_input, sample_kwargs = ema_model.calls[0]
    assert sampled_input is model_input
    assert sample_kwargs == {
        "evaluation_recipe": PROTENIX_BASELINE_RECIPE,
        "mc_dropout": True,
        "mc_dropout_rate": 0.4,
        "tfg_guidance": None,
    }
    assert loss is not None and loss.item() == 3.0
    assert criterion.calls and criterion.calls[0][1] == {
        "input": model_input,
        "target": structure_target,
    }
    assert isinstance(prediction, ProtenixEvaluationOutput)
    torch.testing.assert_close(prediction.coordinates, permutation.coordinates)
    expected_heads = permutation.gather_heads(
        plddt_logits=raw_prediction.plddt_logits,
        resolved_logits=raw_prediction.resolved_logits,
        pae_logits=raw_prediction.pae_logits,
        pde_logits=raw_prediction.pde_logits,
    )
    for name, expected_head in expected_heads.items():
        torch.testing.assert_close(getattr(prediction, name), expected_head)
    torch.testing.assert_close(prediction.distogram_logits, raw_prediction.distogram_logits)
    expected_contact_probs = protenix_contact_probability(raw_prediction.distogram_logits)
    torch.testing.assert_close(prediction.contact_probs, expected_contact_probs)
    torch.testing.assert_close(
        prediction.per_sample_contact_probs,
        permutation.gather_shared_token_pair_axes(expected_contact_probs),
    )
    assert prediction.distogram_logits.shape == (1, 2, 2, 2)
    assert prediction.per_sample_contact_probs.shape == (1, 2, 2, 2)
    assert not torch.equal(
        prediction.per_sample_contact_probs[:, 0],
        prediction.per_sample_contact_probs[:, 1],
    )
    assert runner.meters["objective/mse"].count == 1
    assert runner.meters["objective/weighted_mse"].count == 1
    assert runner.get_step_result()["objective/loss"] == 3.0
    assert len(observed_generic_metrics) == 1
    assert observed_generic_metrics[0][0] is prediction
    assert observed_generic_metrics[0][1] is criterion.calls[0][1]
    assert vars(runner.train_state) == train_state_before
    torch.testing.assert_close(live_model.weight, live_weight)
    torch.testing.assert_close(ema_model.weight, ema_weight)


def test_baseline_evaluate_step_unwraps_live_parallel_model_when_ema_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_prediction = _raw_evaluation_prediction()
    live_model = _SamplingModel(raw_prediction).eval()
    runner = _mock_baseline_evaluation_runner(
        nn.DataParallel(live_model),
        _EvaluationCriterion(),
        ema=None,
    )
    runner._draw_baseline_mc_dropout = lambda: False
    model_input = object()
    structure_target = object()
    identity = ProtenixPredictionPermutation(
        coordinates=raw_prediction.coordinates,
        atom_permutation=torch.arange(3).reshape(1, 1, 3).expand(1, 2, 3),
        token_permutation=torch.arange(2).reshape(1, 1, 2).expand(1, 2, 2),
    )
    monkeypatch.setattr(
        "ofoldx.models.protenix.permutation.match_protenix_prediction",
        lambda coordinates, target, input: identity,
    )

    runner.evaluate_step(
        {
            "input": {"input": model_input, "target": structure_target},
            "target": {"input": model_input, "target": structure_target},
        }
    )

    assert len(live_model.calls) == 1
    assert live_model.calls[0][1]["mc_dropout"] is False


def test_baseline_evaluate_step_rejects_cropped_prediction_axis() -> None:
    atom_count = 8
    full_atom_count = 10
    raw_prediction = _raw_evaluation_prediction(atom_count=atom_count, token_count=atom_count)
    runner = _mock_baseline_evaluation_runner(
        _SamplingModel(raw_prediction).eval(),
        _EvaluationCriterion(),
        ema=None,
    )
    model_input = SimpleNamespace(
        atom_index=torch.arange(atom_count).unsqueeze(0),
        atom_to_token=torch.arange(atom_count).unsqueeze(0),
    )
    structure_target = SimpleNamespace(
        coordinates=torch.zeros(1, full_atom_count, 3),
        coordinate_mask=torch.ones(1, full_atom_count, dtype=torch.bool),
        molecule_id=torch.zeros(1, full_atom_count, dtype=torch.long),
        molecule_equivalence_index=torch.zeros(1, full_atom_count, dtype=torch.long),
        molecule_atom_index=torch.arange(full_atom_count).unsqueeze(0),
        representative_atom_mask=torch.ones(1, full_atom_count, dtype=torch.bool),
        permutation_pool_mask=torch.ones(1, full_atom_count, dtype=torch.bool),
        atom_molecule_type=torch.ones(1, full_atom_count, dtype=torch.long),
        symmetry_atom_index=torch.empty(1, 0, 0, dtype=torch.long),
        symmetry_permutation=torch.empty(1, 0, 0, 0, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="full/no-crop"):
        runner.evaluate_step(
            {
                "input": {"input": model_input, "target": structure_target},
                "target": {"input": model_input, "target": structure_target},
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        {"evaluate_splits": ["val"], "data": {}},
        {"evaluate_splits": [], "data": {"train": {"source": "train.cif"}, "val": {"source": "val.cif"}}},
    ],
)
def test_baseline_evaluation_configuration_reaches_runner_and_data_initialization(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any],
) -> None:
    observed = {"danling": 0, "datasets": 0}

    def fake_danling_init(self: Any, config: RunnerConfig) -> None:
        observed["danling"] += 1
        self.config = config
        self.datasets = {}
        self.meters = AverageMeters()

    def build_datasets(self: Any) -> dict[str, Any]:
        observed["datasets"] += 1
        return {"sentinel": object()}

    monkeypatch.setattr(
        Runner,
        "_resolve_registration",
        lambda self, config, repository: SimpleNamespace(id="protenix"),
    )
    monkeypatch.setattr(
        runner_module.dl.Runner,
        "__init__",
        fake_danling_init,
    )
    monkeypatch.setattr(Runner, "__post_init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture, recipe=OFOLDX_EDM_RECIPE: (
            lambda model, processor, objective_config=None: _FiniteCriterion(),
            _sentinel_collate,
            True,
            True,
        ),
    )
    monkeypatch.setattr(Runner, "_build", lambda self, repository, registration: (_FakeModel(), object()))
    monkeypatch.setattr(Runner, "build_datasets", build_datasets)
    config = {
        "stack": "ddp",
        "model": {"name": "protenix"},
        "logging": {"enabled": False},
        **values,
    }

    runner = Runner(RunnerConfig(config))

    assert observed == {"danling": 1, "datasets": 1}
    assert "sentinel" in runner.datasets


def test_real_danling_state_round_trip_treats_missing_recipe_as_edm_only(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    legacy = _TinyProtenixRunner(_tiny_config(tmp_path / "legacy", OFOLDX_EDM_RECIPE))
    baseline = _TinyProtenixRunner(_tiny_config(tmp_path / "baseline", PROTENIX_BASELINE_RECIPE))
    try:
        legacy.train_state.global_step = 7
        checkpoint = dict(legacy.state_dict())
        checkpoint["runner"] = dict(checkpoint["runner"])
        checkpoint["runner"].pop("objective_recipe")
        legacy.train_state.global_step = 0

        legacy.load_state_dict(checkpoint)
        assert legacy.train_state.global_step == 7
        assert "objective_recipe" not in checkpoint["runner"]

        with pytest.raises(ValueError, match="missing recipe means legacy"):
            baseline.load_state_dict(checkpoint)
    finally:
        baseline.close()
        legacy.close()


def test_real_danling_semantic_check_rejects_explicit_recipe_mismatch(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    legacy = _TinyProtenixRunner(_tiny_config(tmp_path / "legacy", OFOLDX_EDM_RECIPE))
    baseline = _TinyProtenixRunner(_tiny_config(tmp_path / "baseline", PROTENIX_BASELINE_RECIPE))
    try:
        checkpoint = baseline.state_dict()

        with pytest.raises(ValueError, match="semantically different"):
            legacy.load_state_dict(checkpoint)
    finally:
        baseline.close()
        legacy.close()


def test_non_protenix_checkpoint_round_trip_does_not_inject_recipe_or_mutate_caller(
    tmp_path: Path,
    tiny_objective: None,
) -> None:
    source = _TinyProtenixRunner(_tiny_config(tmp_path / "source", None, model_name="proteinmpnn"))
    target = _TinyProtenixRunner(
        _tiny_config(tmp_path / "target", OFOLDX_EDM_RECIPE, model_name="proteinmpnn")
    )
    try:
        source.train_state.global_step = 13
        checkpoint = source.state_dict()
        caller_runner_config = checkpoint["runner"]
        caller_runner_snapshot = dict(caller_runner_config)

        assert "objective_recipe" not in caller_runner_config
        target.load_state_dict(checkpoint)

        assert target.train_state.global_step == 13
        assert checkpoint["runner"] is caller_runner_config
        assert dict(checkpoint["runner"]) == caller_runner_snapshot
        assert "objective_recipe" not in checkpoint["runner"]
    finally:
        target.close()
        source.close()


def test_two_rank_nonfinite_gate_preserves_ddp_lifecycle_and_metric_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")

    class DistributedCriterion:
        def __init__(self) -> None:
            self.calls = 0
            self.latest_metrics: dict[str, torch.Tensor] = {}

        def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss = (prediction - target).square().mean()
            rank = torch.distributed.get_rank()
            if self.calls == 1:
                self.latest_metrics = {"objective/mse": torch.tensor(99.0)}
                if rank == 1:
                    loss = loss * torch.tensor(float("nan"))
            else:
                metric = 1.0 if self.calls == 0 else 4.0
                self.latest_metrics = {"objective/mse": torch.tensor(metric)}
                if self.calls == 2 and rank == 0:
                    self.latest_metrics["objective/plddt"] = torch.tensor(7.0)
            self.calls += 1
            return loss

    def criterion_factory(model: Any, processor: Any, objective_config: Any = None) -> DistributedCriterion:
        del model, processor, objective_config
        return DistributedCriterion()

    monkeypatch.setattr(
        runner_module,
        "_resolve_objective",
        lambda architecture, recipe=OFOLDX_EDM_RECIPE: (
            criterion_factory,
            _sentinel_collate,
            True,
            True,
        ),
    )
    workspace = Path("/tmp") / f"ofoldx-runner-ddp-{os.environ.get('SLURM_JOB_ID', 'local')}"
    config = _tiny_config(workspace, PROTENIX_BASELINE_RECIPE)
    config.update(
        {
            "accum_steps": 2,
            "ema_decay": 0.5,
            "sched": {"type": "step", "step_size": 1, "gamma": 0.9},
            "dist": {"init_timeout_seconds": 60, "train_timeout_seconds": 60},
        }
    )
    runner = _TinyProtenixRunner(config)
    try:
        model = runner.unwrap(runner.model)
        weight_before = model.weight.detach().clone()
        ema_before = runner.ema.state_dict()["weight"].clone()

        rank = torch.distributed.get_rank()
        _, first_loss = runner.train_step(
            {"input": {"value": torch.tensor(float(1 + 2 * rank))}, "target": torch.tensor(0.0)}
        )
        count_before_skip = runner.meters["objective/mse"].count

        _, skipped_loss = runner.train_step(
            {"input": {"value": torch.tensor(2.0)}, "target": torch.tensor(0.0)}
        )
        weight_after_skip = model.weight.detach().clone()
        ema_after_skip = runner.ema.state_dict()["weight"].clone()
        skipped_metrics = runner.get_step_result()
        count_after_skip = runner.meters["objective/mse"].count
        state_after_skip = {
            "weight": weight_after_skip.item(),
            "ema": ema_after_skip.item(),
            "global_step": runner.train_state.global_step,
            "lr": runner.optimizer.param_groups[0]["lr"],
        }
        gathered_after_skip: list[dict[str, Any] | None] = [None, None]
        torch.distributed.all_gather_object(gathered_after_skip, state_after_skip)

        _, third_loss = runner.train_step(
            {"input": {"value": torch.tensor(2.0)}, "target": torch.tensor(0.0)}
        )
        _, fourth_loss = runner.train_step(
            {"input": {"value": torch.tensor(2.0)}, "target": torch.tensor(0.0)}
        )
        epoch_result = runner.get_epoch_result()
        local_state = {
            "weight": model.weight.detach().item(),
            "lr": runner.optimizer.param_groups[0]["lr"],
            "global_step": runner.train_state.global_step,
            "ema": runner.ema.state_dict()["weight"].item(),
        }
        gathered_state: list[dict[str, Any] | None] = [None, None]
        torch.distributed.all_gather_object(gathered_state, local_state)

        assert first_loss is not None and torch.isfinite(first_loss)
        assert skipped_loss is not None and skipped_loss.item() == 0.0
        assert third_loss is not None and torch.isfinite(third_loss)
        assert fourth_loss is not None and torch.isfinite(fourth_loss)
        assert count_before_skip == 1
        assert count_after_skip == count_before_skip
        assert "objective/mse" not in skipped_metrics
        assert gathered_after_skip[0] == gathered_after_skip[1]
        assert state_after_skip["global_step"] == 1
        assert state_after_skip["lr"] == pytest.approx(0.09)
        assert not torch.equal(weight_after_skip, weight_before)
        assert not torch.equal(ema_after_skip, ema_before)
        assert epoch_result["objective/mse"] == pytest.approx(3.0)
        assert epoch_result["objective/plddt"] == pytest.approx(7.0)
        assert gathered_state[0] == gathered_state[1]
        assert local_state["global_step"] == 2
        assert local_state["lr"] == pytest.approx(0.081)
        assert local_state["weight"] != weight_before.item()
        assert local_state["ema"] != ema_before.item()
    finally:
        runner.close()


def test_cli_and_yaml_expose_versioned_objective_recipe(tmp_path: Path) -> None:
    default_config = train.build_config(train.parse_args(["train.cif"]))
    edm_config = train.build_config(
        train.parse_args(["train.cif", "--objective-recipe", OFOLDX_EDM_RECIPE])
    )
    other_config = train.build_config(train.parse_args(["train.cif", "--model", "proteinmpnn"]))

    assert default_config["objective_recipe"] == PROTENIX_BASELINE_RECIPE
    assert edm_config["objective_recipe"] == OFOLDX_EDM_RECIPE
    assert "objective_recipe" not in other_config

    yaml_path = tmp_path / "recipe.yaml"
    yaml_path.write_text(
        f"train_source: train.cif\nobjective_recipe: {OFOLDX_EDM_RECIPE}\n",
        encoding="utf-8",
    )
    yaml_config = train.build_config(train.parse_args(["--config", str(yaml_path)]))
    assert yaml_config["objective_recipe"] == OFOLDX_EDM_RECIPE

    with pytest.raises(SystemExit) as error:
        train.parse_args(
            [
                "train.cif",
                "--model",
                "proteinmpnn",
                "--objective-recipe",
                PROTENIX_BASELINE_RECIPE,
            ]
        )
    assert error.value.code == 2
