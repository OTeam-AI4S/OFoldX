r"""
Training and evaluation runner for the trainable model families.

Trainability is bound to the **exact** registered architecture: an architecture is trainable iff its
[`register`][ofoldx.models.registry.ModelRegistry.register] call declared a training ``criterion`` (and a
``strategy``), verified *before* any model build or weight download. The Runner resolves the architecture
(fresh from ``model.name`` or loaded from ``model.pretrained``), builds the Model and a config-consistent
Processor, then applies the registration's config-aware criterion and one of the generic batching/lowering/
strategies below (the only generic Runner-side training knowledge).

DanLing drives forward → criterion → loss-normalizer → backward → step, and toggles train/eval module mode
itself (train_epoch/evaluate_epoch/infer set ``self.mode``). The Runner adds only what DanLing cannot know
generically: a config-bound criterion, DDP ``find_unused_parameters=True`` (the objective trains only the
loss-path parameters, so auxiliary heads, sampling-only modules, and unexercised conditioning embedders get
no gradient; the models return ``tensorclass`` outputs, which DDP can walk to find them), and an
``infer_step`` that runs the family's inference entry point on its ``tensorclass`` output instead of
DanLing's tensor-shaped default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from tensordict import is_tensorclass

from ..model import Model
from ..models.registry import MODELS, ModelRegistration
from ..processor import Processor
from . import collate as _collate
from .config import RunnerConfig

#: The only DanLing stack whose steps this Runner supports.
_SUPPORTED_STACK = "ddp"


#: Strategy name -> (collate, whether lowering uses the denoising training path).
#: The sequence families collate to a ragged ``NestedTensor`` batch so each structure
#: runs at its own length and padding cannot contaminate k-NN or attention.
#: The diffusion families are batch=1 and lower structures to the ``(input, target)`` the denoising loss
#: needs. The only generic Runner-side training knowledge; a trainable architecture registers a strategy.
_STRATEGIES = {
    "sequence": (_collate.collate_sequence, False),
    "sequence_single": (_collate.collate_sequence_single, False),
    "denoising": (_collate.collate_denoising, True),
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
    Train and evaluate a trainable model family.

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
        self.collate_fn, self._process_training = _STRATEGIES[strategy]
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
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")}
        config = registration.model_config(overrides)
        model = registration.model(config)
        processor = self._build_processor(registration)
        return model, processor

    def _build_processor(self, registration: ModelRegistration) -> Processor:
        r"""
        Build the paired Processor from its own explicit configuration.
        """

        processor = cast("Callable[..., Processor]", registration.processor)
        overrides = dict(self.config.processor)
        config_class = registration.processor.config_class
        if config_class is None:
            if overrides:
                raise ValueError(f"{registration.id} has a configuration-free Processor; drop the processor overrides")
            return processor()
        config = config_class(overrides)
        return processor(config)

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
        Build structure datasets for the configured splits. Every split uses the architecture's training
        lowering: sequence families call the Processor, while diffusion families call ``process_training``
        to produce the separate ``(input, target)`` required by their objective.
        """

        from .dataset import StructureDataset

        datasets: dict[str, Any] = {}
        processor: Any = self.processor
        if self._process_training:
            processor = processor.process_training
        for split, split_config in self.config.data.items():
            processor_kwargs = {key: value for key, value in split_config.items() if key != "source"}
            datasets[split] = StructureDataset(split_config.source, processor, processor_kwargs=processor_kwargs)
        return datasets

    # -- steps ---------------------------------------------------------------

    @torch.inference_mode()
    def infer_step(self, data: Any) -> list[Any]:
        r"""
        Run the family's inference entry point on the whole batch and return one output per structure on the CPU.

        DanLing's default ``infer_step`` calls ``pred.squeeze(-1).tolist()``, which AttributeErrors on the
        family dataclass outputs. This override runs the strategy's inference entry point under DanLing's
        ``infer_context`` -- the plain forward for the sequence families (over a ragged ``NestedTensor`` batch),
        ``model.sample`` for the diffusion families (batch=1). Randomness is seeded from ``config.seed`` and
        the ordinary process-local RNG initialized by DanLing. The ragged output un-nests per structure with
        ``prediction[i]`` -- no padding to trim. Outputs are moved off the accelerator so a long pass does not
        accumulate whole dataclasses on device.

        Decorated with ``@torch.inference_mode()`` like DanLing's default (this override replaces it, so it
        must restore the guard). The model is ``unwrap``-ed first: under DDP ``self.model`` is a
        ``DistributedDataParallel`` wrapper that proxies ``forward`` but not custom methods like ``sample``.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        if model is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        model = self.unwrap(model)  # reach the family's own inference entry point (e.g. ``sample``) under DDP
        model_input = data["input"]
        if self._process_training:
            model_input = model_input["input"]
        with self.infer_context():
            prediction = model.sample(model_input) if self._process_training else model(model_input)
        # One output per structure so infer()'s grouping does not shift with batch size. A ragged tensorclass
        # output un-nests each structure at its own length; a non-tensorclass output passes through.
        if is_tensorclass(prediction) and prediction.batch_size:
            return [dl.to_device(prediction[index], torch.device("cpu")) for index in range(prediction.batch_size[0])]
        return [dl.to_device(prediction, torch.device("cpu"))]
