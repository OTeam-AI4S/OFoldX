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
generically: a config-bound criterion, DDP ``find_unused_parameters=True`` (the objective trains only the
loss-path parameters, so auxiliary heads, sampling-only modules, and unexercised conditioning embedders get
no gradient; the models return ``tensorclass`` outputs, which DDP can walk to find them), a ``tensorclass``-
native ``to_device``, and an ``infer_step`` that runs the family's inference entry point on its
``tensorclass`` output instead of DanLing's tensor-shaped default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from tensordict import is_tensorclass
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


def _trim_tail_padding(output: Any, length: int, padded_length: int) -> Any:
    r"""
    Drop the collate's tail padding from one per-structure sequence Output: keep the first ``length`` rows
    of every residue-aligned tensor field (leading axis == ``padded_length``). This removes only padding, so
    unresolved/gap residues WITHIN ``length`` are preserved -- unlike a ``residue_mask`` index, which would
    also delete those real rows. Non-residue fields pass through unchanged.
    """

    if length == padded_length:
        return output
    kept = {}
    for field in fields(output):
        value = getattr(output, field.name)
        if isinstance(value, Tensor) and value.dim() >= 1 and value.shape[0] == padded_length:
            kept[field.name] = value[:length]
        else:
            kept[field.name] = value
    return type(output)(**kept, batch_size=[])


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
        # Resolve the architecture (no model build) so ``find_unused_parameters`` can be derived from the
        # strategy BEFORE ``super().__init__``. It is a DDP graph-traversal flag, but DanLing's ``config_id``
        # is ``hash(config)`` -- which includes ``ddp`` -- so it must be settled before the identity is
        # frozen. Only the denoising families compute auxiliary heads (confidence, distogram, ...) the
        # objective leaves untrained, so DDP must walk the ``tensorclass`` output to tolerate them; the
        # sequence families train every forward-computed parameter and skip the traversal. An explicit user
        # setting wins (``None`` is "unset").
        registration = self._resolve_registration(config, repository)
        if config.ddp.get("find_unused_parameters") is None:
            config.ddp.find_unused_parameters = registration.strategy == "denoising"
        super().__init__(config)

        self.model, self.processor = self._build(repository, registration)
        criterion, strategy = registration.criterion, registration.strategy
        assert criterion is not None and strategy is not None  # _resolve_registration enforced trainability
        self.criterion = criterion(self.model, self.processor)
        build_collate, self._train_lower, self._infer_lower, self._infer = _STRATEGIES[strategy]
        self.collate_fn = build_collate(registration.collate_options)
        # DanLing toggles module mode from its ``mode`` setter (train_epoch/evaluate_epoch/infer set it), but
        # it fixed ``mode=train`` before this model existed, so the first train_epoch's mode=train is a no-op
        # that will not force train mode -- and a repository model loads in eval(). Establish train mode here.
        self.model.train()
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _resolve_registration(self, config: RunnerConfig, repository: Any) -> ModelRegistration:
        r"""
        Resolve the architecture registration WITHOUT building the model, and confirm it is trainable, so the
        strategy is available before ``super().__init__`` and no download/build happens for an untrainable
        architecture. A repository run resolves from the pinned snapshot metadata and asserts any
        ``model.name`` against it; a fresh run resolves from ``model.name``.
        """

        model_config = config.model
        if model_config.get("pretrained"):
            from ..hub import model_values

            registration, _values = model_values(repository.path)
            if model_config["name"] and model_config["name"] != registration.id:
                raise ValueError(
                    f"config `model.name` {model_config['name']!r} does not match the repository architecture "
                    f"{registration.id!r}"
                )
        else:
            registration = MODELS.lookup_model(model_config["name"])
        _require_trainable(registration)  # fail fast before downloading/building
        return registration

    def _build(self, repository: Any, registration: ModelRegistration) -> tuple[Model, Processor]:
        r"""Instantiate the resolved architecture: load the repository snapshot, or build a fresh pair."""

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(repository)
        overrides = {
            key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")
        }
        model = registration.model(registration.model_config(**overrides))
        processor = self._build_processor(registration, model)
        return model, processor

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

    def _build_from_repository(self, repository: Any) -> tuple[Model, Processor]:
        r"""
        Load a Model/Processor pair from the already-resolved repository snapshot (architecture and
        trainability were settled in ``_resolve_registration``). Load the Processor first -- it validates
        ``processor.json`` from the metadata-only snapshot before ``load_model`` hydrates the full weights.
        """

        from ..hub import load_model, load_processor

        processor = load_processor(repository)
        model = load_model(repository)
        return model, processor

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
                # ``data.to`` already moved tensor fields and nested tensorclasses; only a plain
                # (non-tensorclass) dataclass field -- e.g. Protenix's tfg bundle -- is left behind.
                if is_dataclass(value) and not is_tensorclass(value) and not isinstance(value, type):
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
        model_input = data["input"]  # the infer split collates to ``{"input": model_input}`` (no target)
        with self.infer_context():
            prediction = self._infer(model, model_input)
        # One output per input in the batch, not one per batch, so infer()'s result grouping does not shift
        # with batch size (matching DanLing's per-sample convention). Non-tensorclass outputs pass through.
        if is_tensorclass(prediction) and prediction.batch_size:
            outputs = [prediction[index] for index in range(prediction.batch_size[0])]
            # A batch>1 sequence collate padded every member to the longest; drop each output's tail padding
            # back to its own original length so the result is the same whatever else shared the batch. The
            # collate reports per-item lengths; diffusion (batch=1) reports none, so nothing is trimmed.
            lengths = data.get("lengths") if isinstance(data, Mapping) else None
            if lengths is not None:
                padded_length = max(lengths)
                outputs = [
                    _trim_tail_padding(output, lengths[index], padded_length)
                    for index, output in enumerate(outputs)
                ]
            return [self.to_device(output, torch.device("cpu")) for output in outputs]
        return [self.to_device(prediction, torch.device("cpu"))]
