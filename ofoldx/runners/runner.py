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
CollateBuilder = Callable[[Mapping[str, Any]], Collate]
Infer = Callable[..., Any]


def _decoding_priority(model_input: Any, indices: Any, base_seed: Any, device: Any) -> Any:
    r"""
    Per-structure decoding priority ``(B, N)`` for a ragged sequence batch, one random draw per residue seeded
    by ``base_seed + global_index`` for each structure. Supplying it (rather than letting the model draw from a
    shared generator, whose sequential draws depend on batch order) is what makes a batched forward reproducible
    and independent of batch composition. Returns a ragged ``NestedTensor`` matching the ragged input, or
    ``None`` when there is no seed/index, the batch is not a tensorclass, or the batch is dense (a batch=1
    single-structure family, which has no batch to be invariant to -- ``_forward_infer`` seeds its draw with a
    generator instead).
    """

    from danling import NestedTensor

    if base_seed is None or indices is None or not (is_tensorclass(model_input) and model_input.batch_size):
        return None
    if not isinstance(model_input.sequence_tokens, NestedTensor):
        return None
    priorities = []
    for position, index in enumerate(indices):
        length = int(model_input[position].sequence_tokens.shape[0])
        generator = torch.Generator(device=device).manual_seed(int(base_seed) + int(index))
        priorities.append(torch.rand(length, device=device, generator=generator))
    return NestedTensor(priorities)


def _forward_infer(
    model: Any, model_input: Any, *, indices: Any = None, base_seed: Any = None, device: Any = None
) -> Any:
    r"""Sequence inference is the plain forward over the whole batch. For a ragged batch the Runner supplies a
    per-structure decoding priority seeded by global index; for a dense batch=1 family it seeds the model's own
    draw with a generator (also by global index). Either way each structure's design is reproducible."""

    priority = _decoding_priority(model_input, indices, base_seed, device)
    if priority is not None:
        return model(model_input, decoding_priority=priority)
    generator = None
    if base_seed is not None and indices is not None:
        generator = torch.Generator(device=device).manual_seed(int(base_seed) + int(indices[0]))
    return model(model_input, generator=generator)


def _sample_infer(
    model: Any, model_input: Any, *, indices: Any = None, base_seed: Any = None, device: Any = None
) -> Any:
    r"""Diffusion inference is reverse-diffusion sampling (batch=1); the generator is seeded by the single
    structure's global index so the sample is reproducible and topology-invariant."""

    generator = None
    if base_seed is not None and indices is not None:
        generator = torch.Generator(device=device).manual_seed(int(base_seed) + int(indices[0]))
    return model.sample(model_input, generator=generator)


def _with_indices(collate: Collate) -> Collate:
    r"""
    Wrap a strategy collate so the batch carries each item's global dataset index. The Dataset yields
    ``(index, item)`` pairs; unwrap them, collate the items as usual, and record the indices so
    ``infer_step`` can seed a topology-invariant per-sample generator (``config.seed + index``).
    """

    def collate_with_indices(batch: list[Any]) -> dict[str, Any]:
        result = collate([item for _, item in batch])
        result["indices"] = [index for index, _ in batch]
        return result

    return collate_with_indices


#: Strategy name -> (build the collate from the registration's options, train-split lowering, infer-split
#: lowering, inference entry point). The sequence families collate to a ragged ``NestedTensor`` batch for
#: BOTH training and inference: each structure runs at its own length, so the training loss and inference are
#: topology-invariant (no padding to contaminate the k-NN/attention), and inference un-nests per structure.
#: The diffusion families are batch=1 and lower the training splits to the ``(input, target)`` the denoising
#: loss needs, but the ``infer`` split to the plain Processor call so inference-only kwargs (``tfg_guidance``,
#: ``placement``, ...) flow and no target is built. The only generic Runner-side training knowledge; a
#: trainable arch registers a strategy.
_STRATEGIES: dict[str, tuple[CollateBuilder, str, str, Infer]] = {
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
        self.collate_fn = _with_indices(build_collate(registration.collate_options))
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

        from ..hub import build_registered_config

        model_config = self.config.model
        if model_config.pretrained:
            return self._build_from_repository(repository)
        overrides = {key: value for key, value in model_config.items() if key not in ("name", "pretrained", "revision")}
        config = build_registered_config(registration.model_config, overrides, f"{registration.id} config overrides")
        model = registration.model(config)
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
        the CPU. Family Input/Output/Target are ``tensorclass`` instances, moved with their native ``.to``,
        which also moves nested ``tensorclass`` fields (e.g. Protenix ``template_features``). The one
        tensor-bearing field it cannot reach is a plain (non-``tensorclass``) dataclass -- only Protenix's
        ``tfg_features`` bundle, whose unbatched index tensors cannot be a tensorclass -- and its placement
        is handled by the sampling consumer, so no dataclass recursion is needed here. Mappings/sequences
        recurse; other leaves pass through.
        """

        if device is None:
            device = self.device
        if is_tensorclass(data):
            return data.to(device)
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
        Run the family's inference entry point on the whole batch and return one output per structure on the CPU.

        DanLing's default ``infer_step`` calls ``pred.squeeze(-1).tolist()``, which AttributeErrors on the
        family dataclass outputs. This override runs the strategy's inference entry point under DanLing's
        ``infer_context`` -- the plain forward for the sequence families (over a ragged ``NestedTensor`` batch),
        ``model.sample`` for the diffusion families (batch=1). Randomness is seeded from ``config.seed`` and
        each structure's global dataset index (a per-residue decoding priority for the sequence forward, a
        per-structure generator for diffusion sampling), so every prediction is reproducible and independent of
        batch composition, DDP sharding, and prior RNG history; a ``None`` seed leaves it to ambient RNG. The
        ragged output un-nests per structure with ``prediction[i]`` -- no padding to trim. Outputs are moved off
        the accelerator so a long pass does not accumulate whole dataclasses on device.

        Decorated with ``@torch.inference_mode()`` like DanLing's default (this override replaces it, so it
        must restore the guard). The model is ``unwrap``-ed first: under DDP ``self.model`` is a
        ``DistributedDataParallel`` wrapper that proxies ``forward`` but not custom methods like ``sample``.
        """

        data = self.to_device(data)
        model = self.ema or self.model
        if model is None:
            raise ValueError("cannot run infer_step: model is not initialized")
        model = self.unwrap(model)  # reach the family's own inference entry point (e.g. ``sample``) under DDP
        model_input = data["input"]  # the infer split collates to ``{"input": model_input}`` (no target)
        indices = data.get("indices") if isinstance(data, Mapping) else None
        base_seed = self.config.get("seed")
        with self.infer_context():
            prediction = self._infer(model, model_input, indices=indices, base_seed=base_seed, device=self.device)
        # One output per structure so infer()'s grouping does not shift with batch size. A ragged tensorclass
        # output un-nests each structure at its own length; a non-tensorclass output passes through.
        if is_tensorclass(prediction) and prediction.batch_size:
            return [self.to_device(prediction[index], torch.device("cpu")) for index in range(prediction.batch_size[0])]
        return [self.to_device(prediction, torch.device("cpu"))]
