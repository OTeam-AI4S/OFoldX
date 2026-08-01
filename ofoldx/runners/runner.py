r"""Unified training and evaluation for architectures with grounded objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple, cast

import danling as dl
from danling.data import StepProxyLoader
from torch import Tensor
from torch.utils.data import DistributedSampler

from ..data import Cropper
from ..model import Model
from ..models.registry import MODELS, ModelRegistration
from ..modules.backend import Backend
from ..processor import Processor
from . import collate
from .config import RunnerConfig
from .dataset import SampledStructureDataset, StructureDataset, _collate_with_context
from .sampling import WeightedEpochSampler, read_sampling_index

Collate = Callable[[list[Any]], dict[str, Any]]
Criterion = Callable[[Any, Any], Tensor]


class _TrainingSettings(NamedTuple):
    collate: Collate
    objective: str
    uses_training_processor: bool = False
    find_unused_parameters: bool = False
    forward: Mapping[str, Any] | None = None


def _criterion(objective: str, model: Model, processor: Processor) -> Criterion:
    if objective == "proteinmpnn":
        from ..models.proteinmpnn.objectives import criterion as proteinmpnn_criterion

        return proteinmpnn_criterion
    if objective == "ligandmpnn":
        from ..models.ligandmpnn.objectives import criterion as ligandmpnn_criterion

        return ligandmpnn_criterion
    if objective == "nampnn":
        from ..models.nampnn.objectives import criterion as nampnn_criterion
        from ..models.nampnn.processor import NaMpnnProcessor

        config = cast(NaMpnnProcessor, processor).config
        return partial(nampnn_criterion, shared_nucleic_tokens=config.shared_nucleic_tokens)
    if objective == "grnade":
        from ..models.grnade.objectives import criterion as grnade_criterion

        return grnade_criterion
    if objective == "oinvfold":
        from ..models.oinvfold.objectives import criterion as oinvfold_criterion

        return oinvfold_criterion
    if objective == "alphafold2":
        from ..models.alphafold2.objectives import masked_msa_criterion

        return masked_msa_criterion
    if objective == "denoising":
        from ..modules.diffusion.loss import edm_denoising_loss

        return partial(edm_denoising_loss, sigma_data=float(model.config.denoiser.sigma_data))
    if objective == "boltzgen_ifold":
        from ..models.boltzgen.objectives import residue_type_loss

        def boltzgen_ifold_criterion(prediction: Any, target: Any) -> Tensor:
            return residue_type_loss(
                prediction.logits,
                target["residue_type"],
                target["design_mask"],
            ).mean()

        return boltzgen_ifold_criterion
    if objective == "odesign":
        from ..models.odesign.objectives import odesign_loss

        sigma_data = float(model.config.denoiser.sigma_data)
        num_distogram_bins = int(model.config.pairwise.num_distogram_bins)

        def odesign_criterion(prediction: Any, target: Any) -> Tensor:
            loss, _ = odesign_loss(
                prediction,
                target["input"],
                target["target"],
                sigma_data=sigma_data,
                num_distogram_bins=num_distogram_bins,
            )
            return loss

        return odesign_criterion
    raise RuntimeError(f"unknown Runner objective {objective!r}")


_ALPHAFOLD2_TRAINING = _TrainingSettings(
    collate.collate_alphafold2,
    "alphafold2",
    find_unused_parameters=True,
)
_AF3_TRAINING = _TrainingSettings(
    collate.collate_denoising,
    "denoising",
    uses_training_processor=True,
    find_unused_parameters=True,
    forward={"compute_auxiliary_heads": False},
)
_BOLTZ_TRAINING = _TrainingSettings(
    collate.collate_denoising,
    "denoising",
    uses_training_processor=True,
    find_unused_parameters=True,
)
_BOLTZGEN_TRAINING = _TrainingSettings(
    collate.collate_boltzgen_denoising,
    "denoising",
    uses_training_processor=True,
    find_unused_parameters=True,
)
_ODESIGN_TRAINING = _TrainingSettings(
    collate.collate_odesign,
    "odesign",
    uses_training_processor=True,
)

_TRAINING_SETTINGS = {
    "proteinmpnn": _TrainingSettings(collate.collate_sequence, "proteinmpnn"),
    "nampnn": _TrainingSettings(collate.collate_sequence, "nampnn"),
    "ligandmpnn": _TrainingSettings(collate.collate_sequence_single, "ligandmpnn"),
    "grnade": _TrainingSettings(collate.collate_grnade, "grnade"),
    "grnade_v0_3_2": _TrainingSettings(collate.collate_grnade, "grnade"),
    "oinvfold": _TrainingSettings(collate.collate_oinvfold, "oinvfold"),
    "alphafold2": _ALPHAFOLD2_TRAINING,
    "alphafold2_initial_guess": _ALPHAFOLD2_TRAINING,
    "alphafold2_multimer": _ALPHAFOLD2_TRAINING,
    "alphafold3": _AF3_TRAINING,
    "protenix": _AF3_TRAINING,
    "protenix_template": _AF3_TRAINING,
    "protenix_structure_condition": _AF3_TRAINING,
    "protenix_constraint": _AF3_TRAINING,
    "boltz1": _BOLTZ_TRAINING,
    "boltz2": _BOLTZ_TRAINING,
    "boltzgen": _BOLTZGEN_TRAINING,
    "boltzgen_miniformer": _BOLTZGEN_TRAINING,
    "boltzgen_ifold": _TrainingSettings(
        collate.collate_boltzgen_ifold,
        "boltzgen_ifold",
        uses_training_processor=True,
    ),
    "odesign": _ODESIGN_TRAINING,
    "odesign_ligand": _ODESIGN_TRAINING,
}


def _training_settings(architecture: str) -> _TrainingSettings:
    try:
        return _TRAINING_SETTINGS[architecture]
    except KeyError:
        raise ValueError(f"architecture {architecture!r} has no built-in Runner objective") from None


class Runner(dl.Runner):
    r"""
    Train and evaluate one registered architecture from a single config.

    ProteinMPNN and NA-MPNN collate a ragged batch of any size. Every other objective
    reads an Input whose padded layout is the family's own, which no general rule
    concatenates, so it steps one example at a time and ``dataloader.batch_size`` must
    be 1; raise ``accum_steps`` for a larger effective batch.

    Prediction is a Pipeline's work rather than the Runner's: train or fine-tune here,
    then load the checkpoint with [`ofoldx.pipeline`][].

    Args:
        config: Runner, model, processor, and dataset configuration.
        cropper: Optional structural-token crop policy applied only to training splits.
    """

    config: RunnerConfig
    model: Model
    processor: Processor
    cropper: Cropper | None

    def __new__(
        cls,
        config: RunnerConfig | Mapping[str, Any],
        *,
        cropper: Cropper | None = None,
    ) -> Any:
        r"""Forward only the config through DanLing's stack-selecting constructor."""

        del cropper
        return super().__new__(cls, config)

    def __init__(
        self,
        config: RunnerConfig | Mapping[str, Any],
        *,
        cropper: Cropper | None = None,
    ) -> None:
        if not isinstance(config, RunnerConfig):
            config = RunnerConfig(config)
        config.boot()

        repository = None
        model_config = config.model
        if model_config.get("pretrained"):
            from ..hub import PROCESSOR_METADATA, model_values, resolve_repository

            if dict(config.processor):
                raise ValueError("a repository loads its published Processor config; drop the processor overrides")
            overrides = sorted(key for key in model_config if key not in {"architecture", "pretrained", "revision"})
            if overrides:
                raise ValueError(
                    f"a repository loads its published Model config; drop the model override(s) {overrides}"
                )
            repository = resolve_repository(
                model_config["pretrained"],
                revision=model_config.get("revision"),
                allow_patterns=PROCESSOR_METADATA,
            )
            registration, _ = model_values(repository.path)
            selected = model_config.get("architecture")
            if selected and selected != registration.id:
                raise ValueError(
                    f"model.architecture {selected!r} does not match repository architecture {registration.id!r}"
                )
            if repository.commit:
                config.model.revision = repository.commit
        else:
            registration = MODELS.lookup_model(model_config["architecture"])

        training = _training_settings(registration.id)
        if config.ddp.get("find_unused_parameters") is None:
            config.ddp.find_unused_parameters = training.find_unused_parameters

        super().__init__(config)
        self.model, self.processor = self._build(registration, repository)
        self._validate_model_processor(registration)
        self.model.set_backend(cast(Backend, config.backend))
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.criterion = _criterion(training.objective, self.model, self.processor)
        forward = dict(training.forward or ())
        forward.update(config.forward)
        self.collate_fn = partial(
            _collate_with_context,
            collate_fn=training.collate,
            forward=forward,
        )
        self._training = training
        self.cropper = cropper
        self.model.train()
        self.datasets.update(self.build_datasets())

    def _validate_model_processor(self, registration: ModelRegistration) -> None:
        r"""Validate the model and processor produced by :meth:`_build`."""

        MODELS.for_pair(self.model, self.processor)

    def materialize_model(self) -> None:
        r"""Materialize the root model or its OFoldX-owned compile regions."""

        if self.config.compile.scope == "model" or not self.compiler.enabled:
            super().materialize_model()
            return
        if self.config.stack != "ddp":
            raise NotImplementedError("regional compilation currently requires the ddp runner stack")

        # Regional compilation skips only the root model compile. The caller's
        # DDP policy must still cover DDP construction; PyTorch selects its
        # reducer there, before the regional callables compile lazily.
        with self.compiler.ddp_optimizer():
            compile_enabled = self.compiler.config.enabled
            self.compiler.config.enabled = False
            try:
                super().materialize_model()
            finally:
                self.compiler.config.enabled = compile_enabled

        model = cast(Model, self.unwrap(self.model))
        groups = model.compilation_region_groups()
        if not groups:
            raise ValueError(f"{type(model).__name__} does not define compilation regions")
        for group in groups:
            self.compiler.compile_modules(group)

    def _build(
        self,
        registration: ModelRegistration,
        repository: Any,
    ) -> tuple[Model, Processor]:
        if repository is not None:
            from ..hub import load_model, load_processor

            processor = load_processor(repository)
            model = load_model(repository)
            return model, processor

        values = {
            key: value
            for key, value in self.config.model.items()
            if key not in {"architecture", "pretrained", "revision"}
        }
        model = registration.model(registration.model.config_class(values))
        processor_class = cast(Callable[..., Processor], registration.processor)
        config_class = registration.processor.config_class
        processor_values = dict(self.config.processor)
        if config_class is None:
            if processor_values:
                raise ValueError(f"{registration.id} has a configuration-free Processor; drop the processor overrides")
            processor = processor_class()
        else:
            processor = processor_class(config_class(processor_values))
        return model, processor

    def infer_step(self, data: Any) -> list[Any]:
        r"""
        State that the Runner does not predict.

        A forward call means something different in each family --- logits here, one
        denoising step at a sampled noise level there --- so no single step yields a
        prediction. The Pipeline that owns the family does.
        """

        raise NotImplementedError(
            "the Runner trains and evaluates; load the trained checkpoint with "
            "ofoldx.pipeline(...) and call the family's Pipeline verb to predict"
        )

    def _set_loader_epoch(self, loader: Any, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, WeightedEpochSampler):
            epoch = max(epoch, sampler.epoch)
        super()._set_loader_epoch(loader, epoch)
        if isinstance(loader, StepProxyLoader):
            return
        loader.dataset.set_epoch(epoch)

    def build_datasampler(self, dataset: Any, *, split: str, shuffle: bool) -> Any:
        sampler = super().build_datasampler(dataset, split=split, shuffle=shuffle)
        if not isinstance(dataset, SampledStructureDataset):
            return sampler
        if isinstance(sampler, DistributedSampler):
            num_replicas = sampler.num_replicas
            rank = sampler.rank
        else:
            num_replicas = 1
            rank = 0
        return WeightedEpochSampler(
            dataset.weights,
            dataset.epoch_size,
            replacement=dataset.replacement,
            seed=dataset.sampling_seed,
            num_replicas=num_replicas,
            rank=rank,
        )

    def build_datasets(self) -> dict[str, Any]:
        processor = cast(Callable[..., Any], self.processor)
        if self._training.uses_training_processor:
            processor = cast(Any, self.processor).process_training

        datasets: dict[str, Any] = {}
        for split, split_config in self.config.data.items():
            source = split_config.source
            if source is None:
                raise ValueError(f"data.{split}.source is required")
            sampling = split_config.sampling
            processor_kwargs = dict(split_config.processing)
            dataset_options = {
                "msa_index": split_config.msa_index,
                "msa_root": split_config.msa_root,
                "paired_msa_filename": split_config.paired_msa_filename,
                "unpaired_msa_filename": split_config.unpaired_msa_filename,
            }
            is_train_split = split in self.train_splits or (self.config.train_splits is None and split == "train")
            if self.cropper is not None and is_train_split:
                processor_kwargs = {**processor_kwargs, "cropper": self.cropper}
            if sampling is None:
                datasets[split] = StructureDataset(
                    source,
                    processor,
                    processor_kwargs=processor_kwargs,
                    seed=None if self.config.seed is None else int(self.config.seed),
                    **dataset_options,
                )
                continue
            if not is_train_split:
                raise ValueError("weighted sampling is only available for training splits")
            if not isinstance(sampling, Mapping):
                raise ValueError("data split sampling must be a mapping")
            dataloader = self.config.get("dataloader", {})
            split_dataloader = dataloader.get(split, {})
            if any(dataloader.get(key) is not None for key in ("sampler", "batch_sampler")) or (
                isinstance(split_dataloader, Mapping)
                and any(split_dataloader.get(key) is not None for key in ("sampler", "batch_sampler"))
            ):
                raise ValueError("weighted sampling cannot be combined with a custom sampler or batch_sampler")
            entries = read_sampling_index(sampling["index"])
            datasets[split] = SampledStructureDataset(
                source,
                processor,
                entries,
                epoch_size=sampling.get("epoch_size"),
                replacement=sampling.get("replacement", True),
                sampling_seed=0 if self.config.seed is None else int(self.config.seed),
                processor_kwargs=processor_kwargs,
                seed=None if self.config.seed is None else int(self.config.seed),
                **dataset_options,
            )
        return datasets
