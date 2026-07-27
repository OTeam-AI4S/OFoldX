r"""
Training, evaluation, and inference runner for the trainable model families.

The Runner is config-only and special-cases no family: it resolves an architecture (fresh from
``model.name`` or loaded from ``model.pretrained``), looks up its training pairing in
[`ofoldx.runners.families`][ofoldx.runners.families], and wires the Model, Processor, objective, and
collate. Adding a trainable family is a `families` entry, not a Runner change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

import danling as dl
import torch
from danling.runners.utils import RunnerMode
from torch import Tensor

from ..model import Model
from ..models.registry import MODELS
from ..processor import Processor
from .config import RunnerConfig
from .families import Collate, Objective, build_fresh, wire_loaded
from .registry import RUNNERS


@RUNNERS.register("ofoldx", default=True)
class Runner(dl.Runner):
    r"""
    Train, evaluate, and run forward inference for a trainable model family.

    Args:
        config: Runner configuration, or any mapping it can be built from.
    """

    config: RunnerConfig
    model: Model
    processor: Processor
    objective: Objective
    collate_fn: Collate

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        # Boot (runs ``post`` validation + finalizes) BEFORE DanLing consumes workspace/logging/checkpoint.
        config.boot()
        super().__init__(config)

        self.model, self.processor, self.objective, self.collate_fn = self._build()
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _build(self) -> tuple[Model, Processor, Objective, Collate]:
        r"""
        Resolve the architecture and build a consistent Model, Processor, objective, and collate.
        """

        model_config = self.config.model
        if model_config.pretrained:
            model, processor = self._load_repository(model_config.pretrained)
            registration = MODELS.for_config(model.config)
            if model_config.name and model_config.name != registration.id:
                raise ValueError(
                    f"config `model.name` {model_config.name!r} does not match the loaded repository "
                    f"architecture {registration.id!r}"
                )
            objective, collate = wire_loaded(registration, processor)
            return model, processor, objective, collate
        registration = MODELS.lookup_model(model_config.name)
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained")}
        return build_fresh(registration, overrides)

    @staticmethod
    def _load_repository(pretrained: str) -> tuple[Model, Processor]:
        r"""
        Resolve one repository snapshot and load both the Model and its Processor from it.
        """

        from ..hub import PROCESSOR_METADATA, load_model, load_processor, resolve_repository

        # ``PROCESSOR_METADATA`` covers both ``model.json`` and ``processor.json``, so one resolution
        # feeds both loaders (rather than resolving the repository twice).
        repository = resolve_repository(pretrained, allow_patterns=PROCESSOR_METADATA)
        return load_model(repository), load_processor(repository)

    def build_datasets(self) -> dict[str, Any]:
        r"""
        Build structure datasets for the configured splits, lowered by the paired processor.
        """

        from .dataset import StructureDataset

        datasets: dict[str, Any] = {}
        for split, spec in self.config.data.items():
            processor_kwargs = {key: value for key, value in spec.items() if key != "source"}
            datasets[split] = StructureDataset(spec.source, self.processor, processor_kwargs=processor_kwargs)
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any) -> Any:
        r"""
        Move tensors in a nested batch to the runner device.
        """

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

    # -- steps ---------------------------------------------------------------

    def train_step(self, data: Any) -> tuple[Any, Tensor]:
        r"""
        Run one optimization step.
        """

        data = self.to_device(data)
        with self.train_context():
            output, loss = self.objective(self.model, data)
            self.backward(loss)
            self.step()
        return output, loss

    def evaluate_step(self, data: Any) -> tuple[Any, Tensor]:
        r"""
        Evaluate one batch without parameter updates.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            output, loss = self.objective(model, data)
        return output, loss

    def infer_step(self, data: Any) -> Any:
        r"""
        Run one model forward pass without gradients.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            return model(**data) if isinstance(data, Mapping) else model(data)

    @torch.inference_mode()
    def infer(self, split: str = "infer") -> list[Any]:
        r"""
        Run forward inference over one configured split.

        Raises:
            KeyError: If ``split`` has no dataloader.
        """

        self.mode = RunnerMode.infer
        self.split = split
        model = self.ema or self.model
        model.eval()
        return [self.infer_step(data) for data in self.dataloaders[split]]
