r"""
Training and evaluation runner for the trainable model families.

The Model registry owns only Model/Processor construction identity. The Runner
resolves that pair, then selects its own exact architecture objective and
batching/lowering behavior below. A registered architecture
without a faithful built-in objective remains fully usable for model construction,
loading, and inference; selecting it for this training Runner raises directly.

DanLing drives forward → criterion → loss-normalizer → backward → step, and toggles train/eval module mode
itself (train_epoch/evaluate_epoch/infer set ``self.mode``). The Runner adds only what DanLing cannot know
generically: the architecture's objective and batching, its DDP unused-parameter setting, and an
``infer_step`` that handles structured scientific outputs instead of DanLing's tensor-shaped default.
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


def _resolve_objective(
    architecture: str,
) -> tuple[
    Callable[[Model, Processor], Callable[[Any, Any], torch.Tensor]],
    Callable[[list[Any]], dict[str, Any]],
    bool,
    bool,
]:
    r"""
    Resolve one Runner-owned objective and its batching behavior.

    The booleans independently select ``Processor.process_training`` and DDP
    unused-parameter traversal. Family imports stay local so importing OFoldX
    does not execute every model package.
    """

    if architecture == "proteinmpnn":
        from ..models.proteinmpnn.objectives import criterion_factory

        return criterion_factory, _collate.collate_sequence, False, False
    if architecture == "ligandmpnn":
        from ..models.ligandmpnn.objectives import criterion_factory

        return criterion_factory, _collate.collate_sequence_single, False, False
    if architecture == "nampnn":
        from ..models.nampnn.objectives import criterion_factory

        return criterion_factory, _collate.collate_sequence, False, False
    if architecture in {
        "alphafold3",
        "boltz1",
        "boltz2",
        "boltzgen",
        "protenix",
        "protenix_template",
        "protenix_structure_condition",
        "protenix_constraint",
    }:
        from ..modules.diffusion.loss import denoising_criterion_factory

        return denoising_criterion_factory, _collate.collate_denoising, True, True
    if architecture in {"alphafold2", "alphafold2_initial_guess", "alphafold2_multimer"}:
        from ..models.alphafold2.objectives import criterion_factory

        return criterion_factory, _collate.collate_alphafold2, False, True
    if architecture == "grnade":
        from ..models.grnade.objectives import criterion_factory

        return criterion_factory, _collate.collate_grnade, False, False
    if architecture == "oinvfold":
        from ..models.oinvfold.objectives import criterion_factory

        return criterion_factory, _collate.collate_oinvfold, False, True
    if architecture == "odesign":
        from ..models.odesign.objectives import criterion_factory

        return criterion_factory, _collate.collate_odesign, True, True
    raise ValueError(f"architecture {architecture!r} has no built-in Runner objective")


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
        # Runner-owned training behavior BEFORE ``super().__init__``. It is a DDP graph-traversal flag, but
        # DanLing's ``config_id`` is ``hash(config)`` -- which includes ``ddp`` -- so it must be settled
        # before DanLing freezes the run identity. Architectures with auxiliary heads or alternate decoders
        # need the traversal independently of whether their Processor has a separate training lowering.
        # An explicit user setting wins (``None`` is "unset").
        registration = self._resolve_registration(config, repository)
        criterion_factory, collate_fn, process_training, find_unused_parameters = _resolve_objective(registration.id)
        if config.ddp.get("find_unused_parameters") is None:
            config.ddp.find_unused_parameters = find_unused_parameters
        super().__init__(config)

        self.model, self.processor = self._build(repository, registration)
        self.criterion = criterion_factory(self.model, self.processor)
        self.collate_fn = collate_fn
        self._process_training = process_training
        # DanLing toggles module mode from its ``mode`` setter (train_epoch/evaluate_epoch/infer set it), but
        # it fixed ``mode=train`` before this model existed, so the first train_epoch's mode=train is a no-op
        # that will not force train mode -- and a repository model loads in eval(). Establish train mode here.
        self.model.train()
        self.datasets.update(self.build_datasets())

    # -- construction --------------------------------------------------------

    def _resolve_registration(self, config: RunnerConfig, repository: Any) -> ModelRegistration:
        r"""
        Resolve the architecture registration without building the model. A
        repository run resolves from the pinned snapshot metadata and asserts any
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
        Load a Model/Processor pair from the already-resolved repository snapshot. Load the Processor first --
        it validates
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
        family dataclass outputs. This override runs the family's inference entry point under DanLing's
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
