r"""
Training, evaluation, and inference runner for the trainable model families.

Trainability is bound to the **exact** architecture id through its family's training contract (see
[`TrainingContract`][ofoldx.models.training.TrainingContract]), never guessed from a family directory, and
is verified *before* any model build or weight download. The Runner resolves the architecture (fresh from
``model.name`` or loaded from ``model.pretrained``), builds the Model and a config-consistent Processor, then
applies the contract: its config-aware criterion, one of the generic batching/lowering/inference strategies
below (the only generic Runner-side training knowledge), and freezing of the modules the objective does not
train.

DanLing drives forward → criterion → loss-normalizer → backward → step. The Runner adds only what DanLing
cannot know generically: a config-bound criterion, a train/eval **mode** guard (DanLing never toggles module
mode, and repository models load in ``eval()``), DDP ``find_unused_parameters=True`` (the objective trains
only the loss-path parameters, so auxiliary heads, sampling-only modules, and unexercised conditioning
embedders get no gradient; the models return ``tensorclass`` outputs, which DDP can walk to find them),
a ``tensorclass``-native ``to_device``, and an ``infer_step`` that runs the family's inference entry point
on its ``tensorclass`` output instead of DanLing's tensor-shaped default.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from tensordict import is_tensorclass
from torch import Tensor

from ..model import Model
from ..models.registry import BUILTIN_MODULES, MODELS, ModelRegistration
from ..processor import Processor
from . import collate as _collate
from .config import RunnerConfig

#: The only DanLing stack whose steps this Runner supports.
_SUPPORTED_STACK = "ddp"

Collate = Callable[[list[Any]], dict[str, Any]]
Infer = Callable[[Any, Any], Any]


def _forward_infer(model: Any, model_input: Any) -> Any:
    r"""Sequence inference is the plain forward (logits over the design positions)."""

    return model(model_input)


def _sample_infer(model: Any, model_input: Any) -> Any:
    r"""Diffusion inference is reverse-diffusion sampling (the training forward needs a target)."""

    return model.sample(model_input)


#: Strategy name -> (build a collate from the contract's options, dataset lowering, inference entry point).
#: The only generic Runner-side training knowledge; a family's contract names one of these.
_STRATEGIES: dict[str, tuple[Callable[[Mapping[str, Any]], Collate], str, Infer]] = {
    "sequence": (lambda options: (lambda batch: _collate.collate_sequence(batch, **options)), "call", _forward_infer),
    "sequence_single": (lambda options: _collate.collate_sequence_single, "call", _forward_infer),
    "denoising": (lambda options: _collate.collate_denoising, "process_training", _sample_infer),
}


@dataclass(frozen=True)
class _TrainingSpec:
    r"""A family contract resolved against its strategy: everything the Runner needs to train one model."""

    criterion: Callable[[Model, Processor], Callable[[Any, Any], Any]]
    collate: Collate
    lower: str
    infer: Infer


def _training_spec(architecture: str) -> _TrainingSpec:
    r"""
    Resolve the exact architecture's family-declared training contract into a runnable spec.

    The family's ``objectives`` module exposes a ``TRAINING`` contract listing the exact ids it covers, so
    a same-package variant it does not cover (or an architecture with no ``objectives`` module) is not
    trainable.
    """

    module_path = f"{BUILTIN_MODULES[architecture]}.objectives"
    try:
        objectives = importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name != module_path:
            raise  # a genuinely missing dependency inside objectives, not an absent objectives module
        objectives = None
    contract = getattr(objectives, "TRAINING", None)
    if contract is None or architecture not in contract.architectures:
        raise ValueError(
            f"architecture {architecture!r} is not trainable: its family declares no training contract "
            "covering it"
        )
    strategy = _STRATEGIES.get(contract.strategy)
    if strategy is None:
        raise ValueError(
            f"the training contract for {architecture!r} names unknown strategy {contract.strategy!r}; known "
            f"strategies are {', '.join(sorted(_STRATEGIES))}"
        )
    build_collate, lower, infer = strategy
    return _TrainingSpec(
        criterion=contract.criterion,
        collate=build_collate(contract.collate_options),
        lower=lower,
        infer=infer,
    )


class Runner(dl.Runner):
    r"""
    Train, evaluate, and run inference for a trainable model family.

    Args:
        config: Runner configuration, or any mapping it can be built from.
    """

    config: RunnerConfig
    model: Model
    processor: Processor
    spec: _TrainingSpec

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

        self.spec, self.model, self.processor = self._build()
        self.criterion = self.spec.criterion(self.model, self.processor)
        self.collate_fn = self.spec.collate
        # The objective trains only the loss-path parameters; auxiliary heads / sampling-only modules /
        # unexercised conditioning embedders get no gradient, so DDP must tolerate unused parameters. The
        # models now return ``tensorclass`` outputs, which DDP can walk, so ``find_unused_parameters=True``
        # works (and re-checks each step, unlike ``static_graph``, so it also handles the data-dependent
        # template case).
        self.config.ddp.find_unused_parameters = True
        self.model.train()  # repository models load in eval(); DanLing never toggles module mode
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _build(self) -> tuple[_TrainingSpec, Model, Processor]:
        r"""Resolve the architecture, confirm it is trainable *before* building/loading, then instantiate."""

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(model_config)
        registration = MODELS.lookup_model(model_config["name"])
        spec = _training_spec(registration.id)  # fail fast before building the model
        overrides = {
            key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")
        }
        model = registration.model(registration.model_config(**overrides))
        processor = self._build_processor(registration, model)
        return spec, model, processor

    def _build_processor(self, registration: ModelRegistration, model: Model) -> Processor:
        r"""
        Build the paired Processor: thread the keys it genuinely shares with the Model config (e.g.
        ``ca_only``) from the Model, so one setting drives both. A shared key may not also appear in the
        ``processor`` overrides -- that would let the Processor and Model disagree (an incompatible pair).
        """

        processor = cast("Callable[..., Processor]", registration.processor)
        overrides = dict(self.config.processor)
        config_class = registration.processor.config_class
        if config_class is None:
            if overrides:
                raise ValueError(f"{registration.id} has a configuration-free Processor; drop the processor overrides")
            return processor()
        shared = {key: model.config[key] for key in config_class() if key in model.config}
        conflicting = sorted(shared.keys() & overrides.keys())
        if conflicting:
            raise ValueError(
                f"processor override(s) {conflicting} are shared with the model config; set them on the model "
                "so the Model and Processor stay consistent"
            )
        return processor(config_class(**{**shared, **overrides}))

    def _build_from_repository(self, model_config: Mapping[str, Any]) -> tuple[_TrainingSpec, Model, Processor]:
        r"""
        Load a Model/Processor pair from one repository snapshot: decide architecture + trainability and
        validate the Processor metadata from the snapshot *before* the expensive weight download, then pin
        the resolved commit so a mutable Hub id resumes reproducibly.
        """

        from ..hub import PROCESSOR_METADATA, load_model, load_processor, model_values, resolve_repository

        if dict(self.config.processor):
            raise ValueError("a repository loads its Processor with the published config; drop the processor overrides")
        model_overrides = sorted(k for k in model_config if k not in ("name", "pretrained", "revision"))
        if model_overrides:
            raise ValueError(
                f"a repository loads its Model with the published config; drop the model override(s) {model_overrides}"
            )
        repository = resolve_repository(
            model_config["pretrained"],
            revision=model_config.get("revision"),
            allow_patterns=PROCESSOR_METADATA,
        )
        metadata_registration, _values = model_values(repository.path)
        architecture = metadata_registration.id
        if model_config["name"] and model_config["name"] != architecture:
            raise ValueError(
                f"config `model.name` {model_config['name']!r} does not match the repository architecture "
                f"{architecture!r}"
            )
        spec = _training_spec(architecture)  # fail fast before downloading weights
        # Load the Processor first -- it validates ``processor.json`` from the metadata-only snapshot before
        # ``load_model`` hydrates the full weights.
        processor = load_processor(repository)
        model = load_model(repository)
        if repository.commit:
            self.config.model.revision = repository.commit  # pin the exact snapshot for resume
        return spec, model, processor

    def build_datasets(self) -> dict[str, Any]:
        r"""
        Build structure datasets for the configured splits, all lowered by the spec's lowering (diffusion
        families need the separate ``(input, target)``; the target is unused at inference). ``infer_step``
        takes the model-visible input from the batch, so every split collates the same way.
        """

        from .dataset import StructureDataset

        datasets: dict[str, Any] = {}
        for split, split_config in self.config.data.items():
            processor_kwargs = {key: value for key, value in split_config.items() if key != "source"}
            datasets[split] = StructureDataset(
                split_config.source, self.processor, lower=self.spec.lower, processor_kwargs=processor_kwargs
            )
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any) -> Any:
        r"""Move a nested batch (family Input/Target dataclasses and their nested fields) to the runner device."""

        return self._move(data, self.device)

    def _move(self, data: Any, device: torch.device) -> Any:
        r"""
        Recursively move a nested structure to ``device``. Family Input/Output/Target are ``tensorclass``
        instances -- moved with their native ``.to`` (which recurses into nested tensorclasses and keeps
        ``batch_size``; ``dataclasses.replace`` would reset it). Mappings/sequences recurse; non-tensor
        leaves pass through. The plain-dataclass branch remains for any non-tensorclass dataclass.
        """

        if is_tensorclass(data):
            return data.to(device)
        if is_dataclass(data) and not isinstance(data, type):
            moved = {
                field.name: self._move(value, device)
                for field in fields(data)
                if (value := getattr(data, field.name)) is not None
            }
            return replace(data, **moved)
        if isinstance(data, Mapping):
            moved = {key: self._move(value, device) for key, value in data.items()}
            return type(data)(moved) if isinstance(data, dict) else moved
        if isinstance(data, list):
            return [self._move(value, device) for value in data]
        if isinstance(data, tuple):
            return tuple(self._move(value, device) for value in data)
        if isinstance(data, Tensor):
            return data.to(device)
        return data

    # -- steps ---------------------------------------------------------------

    def train_step(self, data: Any) -> tuple[Any, Any]:
        if not self.model.training:
            self.model.train()
        return super().train_step(data)

    def evaluate_step(self, data: Any) -> tuple[Any, Any]:
        model = self.ema or self.model
        if model.training:
            model.eval()
        return super().evaluate_step(data)

    def infer_step(self, data: Any) -> list[Any]:
        r"""
        Run the family's inference entry point and return its output(s) on the CPU.

        DanLing's default ``infer_step`` calls ``pred.squeeze(-1).tolist()``, which AttributeErrors on the
        family dataclass outputs. This override runs ``spec.infer`` under DanLing's ``infer_context`` -- the
        plain forward for the sequence families, ``model.sample`` for the diffusion families (whose training
        forward needs a target that inference does not have) -- on the model-visible input, then moves the
        output off the accelerator so a long inference pass does not accumulate whole dataclasses on device.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        if model is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        if model.training:
            model.eval()
        inputs = data["input"] if isinstance(data, Mapping) else data
        model_input = inputs["input"] if isinstance(inputs, Mapping) else inputs
        with self.infer_context():
            prediction = self.spec.infer(model, model_input)
        return [self._move(prediction, torch.device("cpu"))]
