r"""
Training, evaluation, and inference runner for the trainable model families.

Trainability is bound to the **exact** registered architecture: an architecture is trainable iff its
[`register`][ofoldx.models.registry.ModelRegistry.register] call declared a training ``criterion`` (and a
``strategy``), verified *before* any model build or weight download. The Runner resolves the architecture
(fresh from ``model.name`` or loaded from ``model.pretrained``), builds the Model and a config-consistent
Processor, then applies the registration's config-aware criterion and one of the generic batching/lowering/
inference strategies below (the only generic Runner-side training knowledge).

DanLing drives forward → criterion → loss-normalizer → backward → step, and toggles train/eval module mode
itself (train_epoch/evaluate_epoch/infer set ``self.mode``). The Runner adds only what DanLing cannot know
generically: a config-bound criterion, DDP ``static_graph`` (the objective trains only the loss-path
parameters, so auxiliary heads, sampling-only modules, and unexercised conditioning embedders get no
gradient; static_graph learns that fixed unused set from the first iteration, so the plain-dataclass forward
outputs need not be a walkable pytree), a nested ``to_device`` for the family dataclasses, and an
``infer_step`` that runs the family's inference entry point on its dataclass output instead of DanLing's
tensor-shaped default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from torch import Tensor

from ..model import Model
from ..models.registry import MODELS, ModelRegistration
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


#: Strategy name -> (build a collate from the registration's options, train-split lowering, infer-split
#: lowering, inference entry point). The diffusion families lower the training splits to the ``(input,
#: target)`` the denoising loss needs, but the ``infer`` split to the plain Processor call so inference-only
#: kwargs (``tfg_guidance``, ``placement``, ...) flow and no target is built. The sequence families lower
#: both the same way. The only generic Runner-side training knowledge; a trainable arch registers a strategy.
_STRATEGIES: dict[str, tuple[Callable[[Mapping[str, Any]], Collate], str, str, Infer]] = {
    "sequence": (
        lambda options: (lambda batch: _collate.collate_sequence(batch, **options)),
        "call",
        "call",
        _forward_infer,
    ),
    "sequence_single": (lambda options: _collate.collate_sequence_single, "call", "call", _forward_infer),
    "denoising": (lambda options: _collate.collate_denoising, "process_training", "call", _sample_infer),
}


def _require_trainable(registration: ModelRegistration) -> None:
    r"""
    Raise unless the architecture is trainable: its registration declares a training ``criterion`` and a
    ``strategy`` the Runner implements. Trainability is a property of the exact registered architecture, so a
    same-family variant that registered no criterion is simply not trainable.
    """

    if registration.criterion is None:
        raise ValueError(
            f"architecture {registration.id!r} is not trainable: its registration declares no training criterion"
        )
    if registration.strategy not in _STRATEGIES:
        raise ValueError(
            f"architecture {registration.id!r} registers unknown training strategy {registration.strategy!r}; "
            f"known strategies are {', '.join(sorted(_STRATEGIES))}"
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
        # The objective trains only the loss-path parameters; auxiliary heads (confidence/distogram/B-factor),
        # sampling-only modules, and unexercised conditioning embedders get no gradient. static_graph learns
        # that fixed unused set from the first iteration's autograd hooks, so DDP tolerates it without a freeze
        # list and without needing the plain-dataclass forward outputs to be a walkable pytree (which
        # find_unused_parameters would require for a computed-but-unused head). Constant unused set holds for
        # the homogeneous, fixed-config training the trainable families use. Set it BEFORE ``super().__init__``
        # so DanLing hashes it into the run identity/workspace.
        config.ddp.static_graph = True
        super().__init__(config)

        self.model, self.processor, registration = self._build(repository)
        criterion, strategy = registration.criterion, registration.strategy
        assert criterion is not None and strategy is not None  # _require_trainable enforced this in _build
        self.criterion = criterion(self.model, self.processor)
        build_collate, self._train_lower, self._infer_lower, self._infer = _STRATEGIES[strategy]
        self.collate_fn = build_collate(registration.collate_options)
        # DanLing toggles module mode from its ``mode`` setter (train_epoch/evaluate_epoch/infer set it), but
        # it fixed ``mode=train`` before this model existed, so the first train_epoch's mode=train is a no-op
        # that will not force train mode -- and a repository model loads in eval(). Establish train mode here.
        self.model.train()
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _build(self, repository: Any = None) -> tuple[Model, Processor, ModelRegistration]:
        r"""Resolve the architecture, confirm it is trainable *before* building/loading, then instantiate."""

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(model_config, repository)
        registration = MODELS.lookup_model(model_config["name"])
        _require_trainable(registration)  # fail fast before building the model
        overrides = {
            key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")
        }
        model = registration.model(registration.model_config(**overrides))
        processor = self._build_processor(registration, model)
        return model, processor, registration

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
    ) -> tuple[Model, Processor, ModelRegistration]:
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
        _require_trainable(metadata_registration)  # fail fast before downloading weights
        # Load the Processor first -- it validates ``processor.json`` from the metadata-only snapshot before
        # ``load_model`` hydrates the full weights.
        processor = load_processor(repository)
        model = load_model(repository)
        return model, processor, metadata_registration

    def build_datasets(self) -> dict[str, Any]:
        r"""
        Build structure datasets for the configured splits. Training and evaluation splits lower to whatever
        the forward + criterion need (the diffusion families' separate ``(input, target)``); the ``infer``
        split -- DanLing's inference split -- lowers with the plain Processor call, so inference-only kwargs
        (``tfg_guidance``, ``placement``, ...) declared beside its ``source`` flow and no target is built.
        """

        from .dataset import StructureDataset

        datasets: dict[str, Any] = {}
        for split, split_config in self.config.data.items():
            processor_kwargs = {key: value for key, value in split_config.items() if key != "source"}
            lower = self._infer_lower if split == "infer" else self._train_lower
            datasets[split] = StructureDataset(
                split_config.source, self.processor, lower=lower, processor_kwargs=processor_kwargs
            )
        return datasets

    # -- batch movement ------------------------------------------------------

    def to_device(self, data: Any, device: torch.device | None = None) -> Any:
        r"""
        Recursively move a nested batch to ``device`` -- the runner device by default; ``infer_step`` passes
        the CPU. Family Input/Output/Target are plain dataclasses: the dataclass branch rebuilds them with
        every field moved, so tensors nested inside a dataclass field (e.g. Protenix's tfg/template feature
        bundles) are moved too. Mappings/sequences recurse; non-tensor leaves pass through.
        """

        if device is None:
            device = self.device
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
        model_input = data["input"]  # the infer split collates to ``{"input": model_input}`` (no target)
        with self.infer_context():
            prediction = self._infer(model, model_input)
        return [self.to_device(prediction, torch.device("cpu"))]
