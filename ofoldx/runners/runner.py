r"""
Training, evaluation, and inference runner for registered OFoldX models.
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
from ..processor import Processor
from .collate import collate
from .config import RunnerConfig
from .registry import ARCHITECTURE_OBJECTIVES, OBJECTIVES, RUNNERS


@RUNNERS.register("ofoldx", default=True)
class Runner(dl.Runner):
    r"""
    Train, evaluate, and run forward inference for a registered model.

    Args:
        config: Runner configuration or values used to construct one.
    """

    config: RunnerConfig

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        super().__init__(config)
        # Config-only construction can skip chanfig's ``boot`` (which runs ``post``), so run the
        # curated ``RunnerConfig.post`` validation here before ``build_model`` -- otherwise a missing
        # ``model.name`` surfaces as ``MODELS.lookup_model(None)``'s opaque "unknown model architecture".
        self.config.post()
        self._architecture = ""
        self.model = self.build_model()
        from .objectives import diffusion, mpnn  # noqa: F401 — register the built-in objectives

        name = self.config.objective or ARCHITECTURE_OBJECTIVES.get(self._architecture)
        if name is None:
            raise ValueError(
                f"no training objective is defined for architecture {self._architecture!r}; "
                f"set `objective` explicitly or use an architecture with a registered objective "
                f"({', '.join(sorted(set(ARCHITECTURE_OBJECTIVES.values())))})"
            )
        self.objective = OBJECTIVES.build(type=name)
        self.collate_fn = collate
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def build_model(self) -> Model:
        r"""
        Build the configured model.

        Returns:
            A pretrained model when ``model.pretrained`` is set, otherwise a model
            initialized from the registered architecture config.
        """

        from ..models.registry import MODELS

        model_config = self.config.model
        registration = MODELS.lookup_model(model_config.name)
        self._architecture = registration.id
        if model_config.pretrained:
            from ..auto import AutoModel

            return AutoModel(model_config.pretrained)
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained")}
        typed_config = registration.model_config(**overrides) if overrides else registration.model_config()
        return registration.model(typed_config)

    def build_datasets(self) -> dict[str, Any]:
        r"""
        Build datasets for configured splits.

        Returns:
            Structure datasets keyed by split name.
        """

        from ..models.registry import MODELS
        from .dataset import StructureDataset

        model_config = self.config.model
        processor: Processor
        if model_config.pretrained:
            from ..auto import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_config.pretrained)
        else:
            processor = MODELS.lookup_model(model_config.name).processor()
        datasets: dict[str, Any] = {}
        for split, spec in self.config.data.items():
            processor_kwargs = {key: value for key, value in spec.items() if key != "source"}
            datasets[split] = StructureDataset(spec.source, processor, processor_kwargs=processor_kwargs)
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any) -> Any:
        r"""
        Move tensors in a nested batch to the runner device.

        Args:
            data: Tensor, dataclass, mapping, or sequence batch.

        Returns:
            The batch with tensor values moved to the runner device.
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

        Args:
            data: Batch accepted by the configured objective.

        Returns:
            The model output and scalar training loss.
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

        Args:
            data: Batch accepted by the configured objective.

        Returns:
            The model output and scalar evaluation loss.
        """
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            output, loss = self.objective(model, data)
        return output, loss

    def infer_step(self, data: Any) -> Any:
        r"""
        Run one model forward pass without gradients.

        Args:
            data: Positional model input or mapping of model arguments.

        Returns:
            The model output.
        """
        data = self.to_device(data)
        model = self.ema or self.model
        with self.infer_context():
            return model(**data) if isinstance(data, Mapping) else model(data)

    @torch.inference_mode()
    def infer(self, split: str = "infer") -> list[Any]:
        r"""
        Run forward inference over one configured split.

        Args:
            split: Dataset split to evaluate.

        Returns:
            Model outputs in dataloader order.

        Raises:
            KeyError: If ``split`` has no dataloader.
        """

        self.mode = RunnerMode.infer
        self.split = split
        model = self.ema or self.model
        model.eval()
        return [self.infer_step(data) for data in self.dataloaders[split]]
