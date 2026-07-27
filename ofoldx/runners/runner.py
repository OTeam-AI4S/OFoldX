r"""
Training, evaluation, and inference runner for the trainable model families.

The Runner special-cases no family and adds almost nothing to DanLing. It resolves an architecture
(fresh from ``model.name`` or loaded from ``model.pretrained``), builds the Model and a config-consistent
Processor, and discovers the family's training contract *by convention*: the family package exports a
DanLing-shaped ``criterion(pred, target) -> loss`` and a ``collate`` that produces DanLing's
``{"input", "target", "loss_normalizer"}`` batch. Everything else -- forward, criterion, the
per-token loss normalizer, backward, accumulation -- is DanLing's own ``train_step``. A family with no
``criterion`` is simply not trainable.

V1 targets the plain Torch/DDP stack; the only overrides below are a train/eval **mode** guard, because
DanLing never toggles module ``train()``/``eval()`` and repository models load in ``eval()``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import danling as dl
from danling.runners.config import normalize_stack_name
from torch import Tensor

from ..model import Model
from ..models.registry import BUILTIN_MODULES, MODELS, ModelRegistration
from ..processor import Processor
from .config import RunnerConfig
from .registry import RUNNERS

#: The only DanLing stack whose steps this Runner supports.
_SUPPORTED_STACK = "ddp"

Criterion = Callable[[Any, Any], Tensor]
Collate = Callable[[list[Any]], Any]


def _training_contract(architecture: str) -> tuple[Criterion, Collate]:
    r"""
    Discover a family's training contract from its package (``criterion`` + ``collate``).
    """

    module = importlib.import_module(BUILTIN_MODULES[architecture])
    if not hasattr(module, "criterion"):
        raise ValueError(
            f"architecture {architecture!r} is not trainable: {BUILTIN_MODULES[architecture]} exports no "
            "`criterion(pred, target) -> loss`"
        )
    return module.criterion, module.collate


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

    def __init__(self, config: RunnerConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        # Boot (runs ``post`` + finalizes) BEFORE DanLing consumes workspace/logging/checkpoint.
        config.boot()
        stack = normalize_stack_name(config.get("stack", "auto"))
        if stack != _SUPPORTED_STACK:
            raise ValueError(
                f"the OFoldX Runner supports only the plain Torch/DDP stack ({_SUPPORTED_STACK!r}); got "
                f"{stack!r}. Graph and parallel stacks require their own steps."
            )
        super().__init__(config)

        self.model, self.processor, self.criterion, self.collate_fn = self._build()
        self.model.train()  # repository models load in eval(); DanLing never toggles module mode
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _build(self) -> tuple[Model, Processor, Criterion, Collate]:
        r"""
        Resolve the architecture, confirm it is trainable *before* building/loading, then instantiate.
        """

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(model_config)
        registration = MODELS.lookup_model(model_config["name"])
        criterion, collate = _training_contract(registration.id)  # fail fast before building the model
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained")}
        model = registration.model(registration.model_config(**overrides))
        processor = self._build_processor(registration, model)
        return model, processor, criterion, collate

    def _build_processor(self, registration: ModelRegistration, model: Model) -> Processor:
        r"""
        Build the paired Processor: thread the keys it genuinely shares with the Model config (e.g.
        ``ca_only``) from the Model, so one setting drives both, then apply processor-only overrides.
        A configuration-free Processor takes neither.
        """

        processor = cast("Callable[..., Processor]", registration.processor)
        config_class = registration.processor.config_class
        if config_class is None:
            return processor()
        shared = {key: model.config[key] for key in config_class() if key in model.config}
        return processor(config_class(**{**shared, **dict(self.config.processor)}))

    def _build_from_repository(self, model_config: Mapping[str, Any]) -> tuple[Model, Processor, Criterion, Collate]:
        r"""
        Load a Model/Processor pair from one repository snapshot, deciding architecture + trainability
        from the metadata before any weights download.
        """

        from ..hub import PROCESSOR_METADATA, load_model, load_processor, model_values, resolve_repository

        # ``PROCESSOR_METADATA`` covers both ``model.json`` and ``processor.json``, so one resolution
        # feeds both loaders and the metadata check.
        repository = resolve_repository(model_config["pretrained"], allow_patterns=PROCESSOR_METADATA)
        metadata_registration, _values = model_values(repository.path)
        architecture = metadata_registration.id
        if model_config["name"] and model_config["name"] != architecture:
            raise ValueError(
                f"config `model.name` {model_config['name']!r} does not match the repository architecture "
                f"{architecture!r}"
            )
        criterion, collate = _training_contract(architecture)  # fail fast before downloading weights
        model = load_model(repository)
        processor = load_processor(repository)
        return model, processor, criterion, collate

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
        Move tensors in a nested batch (including family Input dataclasses) to the runner device.
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

    # -- steps: DanLing's defaults, plus a train/eval mode guard --------------

    def train_step(self, data: Any) -> tuple[Any, Tensor | None]:
        if not self.model.training:
            self.model.train()
        return super().train_step(data)

    def evaluate_step(self, data: Any) -> tuple[Any, Tensor | None]:
        model = self.ema or self.model
        if model.training:
            model.eval()
        return super().evaluate_step(data)
