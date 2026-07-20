from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import danling
import torch
from torch import Tensor, nn
from torchdata.stateful_dataloader import StatefulDataLoader

import ofoldx
from ofoldx.runners import Runner


@dataclass(frozen=True)
class Layout:
    static_tensor: Tensor


@dataclass(frozen=True)
class Batch:
    inputs: Tensor
    targets: Tensor
    layout: Layout


def collate(batch: list[Batch]) -> Batch:
    return batch[0]


def objective(model: nn.Module, batch: Batch) -> tuple[Tensor, Tensor]:
    output = model(batch.inputs)
    return output, (output - batch.targets).square().mean()


def config(root: Path) -> dict:
    return {
        "logging": {"enabled": False},
        "workspace": {"root": str(root)},
        "epochs": 1,
        "optim": {"type": "sgd", "lr": 0.1, "momentum": 0.9},
        "dataloader": {"batch_size": 1, "shuffle": False, "drop_last": False},
        "ckpt": {"async_mode": "disabled"},
    }


def batch() -> Batch:
    return Batch(torch.ones(1, 1), torch.ones(1, 1), Layout(torch.ones(1)))


def test_runner_uses_danling_dynamic_runtime(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    runner = Runner(
        config(tmp_path),
        model=model,
        datasets={"train": [batch()]},
        train_objective=objective,
        collate_fn=collate,
    )
    try:
        assert ofoldx.Runner is Runner
        assert isinstance(runner, danling.TorchRunner)
        assert runner.model is model
        assert isinstance(runner.dataloaders["train"], StatefulDataLoader)
    finally:
        runner.close()


def test_runner_moves_direct_dataclass_tensors_and_preserves_layout(tmp_path: Path) -> None:
    runner = Runner(config(tmp_path), model=nn.Linear(1, 1), datasets={}, train_objective=objective)
    try:
        original = batch()
        type(runner).device = property(lambda _: torch.device("meta"))
        moved = runner.to_device({"batch": original, "nested": [torch.ones(1)]})

        assert moved["batch"].inputs.device.type == "meta"
        assert moved["batch"].targets.device.type == "meta"
        assert moved["batch"].layout is original.layout
        assert moved["batch"].layout.static_tensor.device.type == "cpu"
        assert moved["nested"][0].device.type == "meta"
    finally:
        runner.close()


def test_runner_train_evaluate_and_infer(tmp_path: Path) -> None:
    calls: list[str] = []

    def train_objective(model: nn.Module, data: Batch) -> tuple[Tensor, Tensor]:
        calls.append("train")
        return objective(model, data)

    def evaluate_objective(model: nn.Module, data: Batch) -> tuple[Tensor, Tensor]:
        calls.append("evaluate")
        return objective(model, data)

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    runner = Runner(
        config(tmp_path) | {"evaluate_splits": ["validation"]},
        model=model,
        datasets={"train": [batch()], "validation": [batch()], "infer": [torch.ones(1, 1)]},
        train_objective=train_objective,
        evaluate_objective=evaluate_objective,
        collate_fn=collate,
    )
    before = model.weight.detach().clone()
    try:
        runner.train()
        outputs = runner.infer()

        assert calls == ["train", "evaluate"]
        assert runner.train_state.global_step == 1
        assert not torch.equal(model.weight, before)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 1)
    finally:
        runner.close()


def test_runner_restores_danling_checkpoint(tmp_path: Path) -> None:
    runner_config = config(tmp_path)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    runner = Runner(
        runner_config,
        model=model,
        datasets={"train": [batch()]},
        train_objective=objective,
        collate_fn=collate,
    )
    try:
        runner.train()
        expected_weight = model.weight.detach().clone()
        checkpoint = Path(runner.workspace.checkpoint_dir) / "latest.pth"
    finally:
        runner.close()

    restored_model = nn.Linear(1, 1, bias=False)
    restored = Runner(
        runner_config | {"checkpoint": str(checkpoint)},
        model=restored_model,
        datasets={"train": [batch()]},
        train_objective=objective,
        collate_fn=collate,
    )
    try:
        torch.testing.assert_close(restored_model.weight, expected_weight)
        assert restored.train_state.global_step == 1
        assert restored.optimizer is not None
        assert restored.optimizer.state_dict()["state"]
    finally:
        restored.close()
