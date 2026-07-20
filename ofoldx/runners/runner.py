r"""Unified Danling Runner for OFoldX models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

import danling
import torch
from danling.runners.utils import RunnerMode
from torch import Tensor, nn
from torch.utils.data import Dataset


class Runner(danling.Runner):
    r"""Train, evaluate, and infer with family-owned computations."""

    def __init__(
        self,
        config: danling.RunnerConfig | Mapping[str, Any],
        *,
        model: nn.Module,
        datasets: Mapping[str, Dataset],
        train_objective: Callable[[nn.Module, Any], tuple[Any, Tensor]],
        evaluate_objective: Callable[[nn.Module, Any], tuple[Any, Tensor]] | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.datasets.update(datasets)
        self.train_objective = train_objective
        self.evaluate_objective = train_objective if evaluate_objective is None else evaluate_objective
        if collate_fn is not None:
            self.collate_fn = collate_fn

    def to_device(self, data: Any) -> Any:
        r"""Move batch tensors while preserving family-owned semantic layout."""

        if is_dataclass(data) and not isinstance(data, type):
            moved = {
                field.name: value.to(self.device)
                for field in fields(data)
                if isinstance(value := getattr(data, field.name), Tensor)
            }
            return replace(data, **moved)
        if isinstance(data, Mapping):
            moved = {key: self.to_device(value) for key, value in data.items()}
            return type(data)(moved) if isinstance(data, dict) else moved
        if isinstance(data, list):
            return [self.to_device(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self.to_device(value) for value in data)
        return super().to_device(data)

    def train_step(self, data: Any) -> tuple[Any, Tensor]:
        data = self.to_device(data)
        with self.train_context():
            output, loss = self.train_objective(self.model, data)
            self.backward(loss)
            self.step()
        return output, loss

    def evaluate_step(self, data: Any) -> tuple[Any, Tensor]:
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            output, loss = self.evaluate_objective(model, data)
        return output, loss

    def infer_step(self, data: Any) -> Any:
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            return model(**data) if isinstance(data, Mapping) else model(data)

    @torch.inference_mode()
    def infer(self, split: str = "infer") -> list[Any]:
        r"""Return generation-ordered Model outputs for one configured split."""

        self.mode = RunnerMode.infer
        self.split = split
        model = self.ema or self.model
        model.eval()
        return [self.infer_step(data) for data in self.dataloaders[split]]
