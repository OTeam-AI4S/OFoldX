r"""The unified config-only Danling Runner for OFoldX Models.

Construction is config-only (DESIGN §10.3): the Runner takes a
:class:`~ofoldx.runners.config.RunnerConfig` and builds the Model, Processor,
datasets, objective, and collate from it, then hands optimizer / scheduler /
dataloader / checkpoint building to danling. Family Models stay
inference-and-forward-only; the training loss is produced by the resolved
:class:`~ofoldx.runners.objectives.base.Objective`, never by the Model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

import danling as dl
import torch
from danling.runners.utils import RunnerMode
from torch import Tensor, nn

from .config import RunnerConfig
from .registry import RUNNERS, build_collate, build_objective


@RUNNERS.register("ofoldx", default=True)
class Runner(dl.Runner):
    r"""Train, evaluate, and infer with any registered OFoldX Model family."""

    config: RunnerConfig

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        super().__init__(config)
        self._architecture: str = ""
        self._processor: Any = None
        self.model = self.build_model()
        self.objective = build_objective(self._architecture, self.config)
        self.collate_fn = build_collate(self._architecture, self.config)
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def build_model(self) -> nn.Module:
        r"""Build the Model from ``config.model`` and record its architecture id.

        ``config.model.pretrained`` loads a registered Model with weights;
        otherwise the Model is randomly initialized from ``config.model.name``
        plus the remaining ``config.model`` keys (typed-config overrides).
        """

        from ..models.registry import _registration

        model_config = self.config.model
        registration = _registration(model_config.name)
        self._architecture = registration.id
        if model_config.pretrained:
            from ..auto import AutoModel

            return AutoModel(model_config.pretrained)
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained")}
        typed_config = registration.model_config(**overrides) if overrides else registration.model_config()
        return registration.model(typed_config)

    def build_processor(self) -> Any:
        r"""Build the family Processor that lowers raw structures into Model inputs."""

        if self._processor is None:
            from ..models.registry import _registration

            model_config = self.config.model
            if model_config.pretrained:
                from ..auto import AutoProcessor

                self._processor = AutoProcessor(model_config.pretrained)
            else:
                self._processor = _registration(model_config.name).processor()
        return self._processor

    def build_datasets(self) -> dict[str, Any]:
        r"""Build one :class:`~ofoldx.runners.dataset.StructureDataset` per configured split."""

        from .dataset import StructureDataset

        processor = self.build_processor()
        datasets: dict[str, Any] = {}
        for split, spec in self.config.data.items():
            processor_kwargs = {key: value for key, value in spec.items() if key not in ("source", "format")}
            datasets[split] = StructureDataset(
                spec.source,
                processor,
                format=spec.format,
                processor_kwargs=processor_kwargs,
            )
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any) -> Any:
        r"""Move batch tensors while preserving the family Input dataclass layout."""

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
        data = self.to_device(data)
        with self.train_context():
            output, loss = self.objective(self.model, data)
            self.backward(loss)
            self.step()
        return output, loss

    def evaluate_step(self, data: Any) -> tuple[Any, Tensor]:
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            output, loss = self.objective(model, data)
        return output, loss

    def infer_step(self, data: Any) -> Any:
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            return model(**data) if isinstance(data, Mapping) else model(data)

    @torch.inference_mode()
    def infer(self, split: str = "infer") -> list[Any]:
        r"""Return the family forward outputs for one configured split.

        This drives the Model's forward graph in batches; the Result-producing
        inference surface (``processor -> model.fold -> FoldResult``) remains the
        Pipelines' responsibility.
        """

        self.mode = RunnerMode.infer
        self.split = split
        model = self.ema or self.model
        model.eval()
        return [self.infer_step(data) for data in self.dataloaders[split]]
