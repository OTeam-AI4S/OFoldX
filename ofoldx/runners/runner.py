r"""
Training, evaluation, and inference runner for the trainable model families.

Trainability is bound to the **exact** architecture id through its family's training contract (see
[`TrainingContract`][ofoldx.models.training.TrainingContract]), never guessed from a family directory, and
is verified *before* any model build or weight download. The Runner resolves the architecture (fresh from
``model.name`` or loaded from ``model.pretrained``), builds the Model and a config-consistent Processor, then
applies the contract: its config-aware criterion, one of the generic batching/lowering/inference strategies
below (the only generic Runner-side training knowledge), and freezing of the modules the objective does not
train.

DanLing drives forward → criterion → loss-normalizer → backward → step, and toggles train/eval module mode
itself (train_epoch/evaluate_epoch/infer set ``self.mode``). The Runner adds only what DanLing cannot know
generically: a config-bound criterion, DDP ``find_unused_parameters=True`` (the objective trains only the
loss-path parameters, so auxiliary heads, sampling-only modules, and unexercised conditioning embedders get
no gradient; the models return ``tensorclass`` outputs, which DDP can walk to find them), a ``tensorclass``-
native ``to_device``, and an ``infer_step`` that runs the family's inference entry point on its
``tensorclass`` output instead of DanLing's tensor-shaped default.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from tensordict import is_tensorclass
from torch import Tensor

from ..model import Model
from ..models.registry import BUILTIN_MODULES, MODELS, ModelRegistration
from ..models.training import TrainingContract
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


def _training_contract(architecture: str) -> TrainingContract:
    r"""
    Return the family-declared training contract that covers this exact architecture, or raise.

    The family's ``objectives`` module exposes a ``TRAINING`` contract listing the exact ids it covers, so
    a same-package variant it does not cover (or an architecture with no ``objectives`` module) is not
    trainable. A contract naming a strategy the Runner does not implement is rejected here too.
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
    if contract.strategy not in _STRATEGIES:
        raise ValueError(
            f"the training contract for {architecture!r} names unknown strategy {contract.strategy!r}; known "
            f"strategies are {', '.join(sorted(_STRATEGIES))}"
        )
    return contract


class Runner(dl.Runner):
    r"""
    Train, evaluate, and run inference for a trainable model family.

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
        # A repository run pins its resolved commit into the config *before* ``super().__init__`` derives the
        # run identity (DanLing freezes ``workspace``/``name`` there and hashes the config for ``id``). Pinning
        # it afterwards would let a mutable branch/tag that later moves resume under an identity that no longer
        # matches the snapshot actually loaded.
        repository = self._prepare_repository(config) if config.model.get("pretrained") else None
        super().__init__(config)

        self.model, self.processor, contract = self._build(repository)
        self.criterion = contract.criterion(self.model, self.processor)
        build_collate, self._lower, self._infer = _STRATEGIES[contract.strategy]
        self.collate_fn = build_collate(contract.collate_options)
        # The objective trains only the loss-path parameters; auxiliary heads / sampling-only modules /
        # unexercised conditioning embedders get no gradient, so DDP must tolerate unused parameters. The
        # models now return ``tensorclass`` outputs, which DDP can walk, so ``find_unused_parameters=True``
        # works (and re-checks each step, unlike ``static_graph``, so it also handles the data-dependent
        # template case).
        self.config.ddp.find_unused_parameters = True
        # DanLing toggles module mode from its ``mode`` setter (train_epoch/evaluate_epoch/infer set it), but
        # it fixed ``mode=train`` before this model existed, so the first train_epoch's mode=train is a no-op
        # that will not force train mode -- and a repository model loads in eval(). Establish train mode here.
        self.model.train()
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _build(self, repository: Any = None) -> tuple[Model, Processor, TrainingContract]:
        r"""Resolve the architecture, confirm it is trainable *before* building/loading, then instantiate."""

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(model_config, repository)
        registration = MODELS.lookup_model(model_config["name"])
        contract = _training_contract(registration.id)  # fail fast before building the model
        overrides = {
            key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")
        }
        # A variant whose canonical fresh graph differs from the base config default (e.g. protenix_template's
        # template embedder) supplies its own config factory; otherwise the registered config_class applies.
        build_config = contract.fresh_configs.get(registration.id) or registration.model_config
        model = registration.model(build_config(**overrides))
        processor = self._build_processor(registration, model)
        return model, processor, contract

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

    def _prepare_repository(self, config: RunnerConfig) -> Any:
        r"""
        Resolve the repository's exact commit and pin it into the config *before* DanLing freezes the run
        identity, rejecting overrides a published pair cannot honor. Runs only for a repository run and
        returns the resolved repository so ``_build_from_repository`` reuses it without a second resolution.
        """

        from ..hub import PROCESSOR_METADATA, resolve_repository

        model_config = config.model
        if dict(config.processor):
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
        if repository.commit:
            config.model.revision = repository.commit  # pin the exact snapshot BEFORE the identity is frozen
        return repository

    def _build_from_repository(
        self, model_config: Mapping[str, Any], repository: Any
    ) -> tuple[Model, Processor, TrainingContract]:
        r"""
        Load a Model/Processor pair from the already-resolved repository snapshot: decide architecture +
        trainability and validate the Processor metadata from the snapshot *before* the expensive weight
        download. ``_prepare_repository`` has already resolved the snapshot and pinned its commit.
        """

        from ..hub import load_model, load_processor, model_values

        metadata_registration, _values = model_values(repository.path)
        architecture = metadata_registration.id
        if model_config["name"] and model_config["name"] != architecture:
            raise ValueError(
                f"config `model.name` {model_config['name']!r} does not match the repository architecture "
                f"{architecture!r}"
            )
        contract = _training_contract(architecture)  # fail fast before downloading weights
        # Load the Processor first -- it validates ``processor.json`` from the metadata-only snapshot before
        # ``load_model`` hydrates the full weights.
        processor = load_processor(repository)
        model = load_model(repository)
        return model, processor, contract

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
                split_config.source, self.processor, lower=self._lower, processor_kwargs=processor_kwargs
            )
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any, device: torch.device | None = None) -> Any:
        r"""
        Recursively move a nested batch to ``device`` -- the runner device by default; ``infer_step`` passes
        the CPU. Family Input/Output/Target are ``tensorclass`` instances, moved with their native ``.to``
        (which keeps ``batch_size``; ``dataclasses.replace`` would reset it). ``.to`` cannot reach tensors
        held inside a non-tensor dataclass field (e.g. Protenix's tfg/template feature bundles, whose index
        tensors are unbatched so they cannot themselves be a tensorclass), so those are moved explicitly.
        Mappings/sequences recurse; non-tensor leaves pass through; the plain-dataclass branch handles any
        non-tensorclass dataclass.
        """

        if device is None:
            device = self.device
        if is_tensorclass(data):
            moved = data.to(device)
            for field in fields(data):
                value = getattr(data, field.name)
                if is_dataclass(value) and not isinstance(value, type):
                    setattr(moved, field.name, self.to_device(value, device))
            return moved
        if is_dataclass(data) and not isinstance(data, type):
            moved = {
                field.name: self.to_device(value, device)
                for field in fields(data)
                if (value := getattr(data, field.name)) is not None
            }
            return replace(data, **moved)
        if isinstance(data, Mapping):
            moved = {key: self.to_device(value, device) for key, value in data.items()}
            return type(data)(moved) if isinstance(data, dict) else moved
        if isinstance(data, list):
            return [self.to_device(value, device) for value in data]
        if isinstance(data, tuple):
            return tuple(self.to_device(value, device) for value in data)
        if isinstance(data, Tensor):
            return data.to(device)
        return data

    # -- steps ---------------------------------------------------------------

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[Any]:
        r"""
        Run the family's inference entry point and return its output(s) on the CPU.

        DanLing's default ``infer_step`` calls ``pred.squeeze(-1).tolist()``, which AttributeErrors on the
        family dataclass outputs. This override runs the strategy's inference entry point under DanLing's
        ``infer_context`` -- the plain forward for the sequence families, ``model.sample`` for the diffusion
        families (whose training forward needs a target that inference does not have) -- on the model-visible
        input, then moves the output off the accelerator so a long inference pass does not accumulate whole
        dataclasses on device.

        Decorated with ``@torch.inference_mode()`` like DanLing's default ``infer_step`` (this override
        replaces it, so it must restore the guard) -- otherwise the forward would build and retain an
        autograd graph for the whole pass. The model is ``unwrap``-ed first: under DDP ``self.model`` is a
        ``DistributedDataParallel`` wrapper that proxies ``forward`` but not custom methods like ``sample``.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        if model is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        model = self.unwrap(model)  # reach the family's own inference entry point (e.g. ``sample``) under DDP
        inputs = data["input"] if isinstance(data, Mapping) else data
        model_input = inputs["input"] if isinstance(inputs, Mapping) else inputs
        with self.infer_context():
            prediction = self._infer(model, model_input)
        return [self.to_device(prediction, torch.device("cpu"))]
