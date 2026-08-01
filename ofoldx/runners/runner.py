# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""Unified training and evaluation for architectures with grounded objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal, NamedTuple, cast

import danling as dl
import torch
from danling.data import StepProxyLoader
from danling.tensors import NestedTensor
from torch import Tensor
from torch.utils.data import DistributedSampler

from ..data.crop import Cropper
from ..data.structure import Structure
from ..data.tokenization import TOKENS
from ..errors import RequestRef, fail
from ..hub import PROCESSOR_METADATA, load_model, load_processor, model_values, resolve_repository
from ..model import Model
from ..models.alphafold2.objective import AlphaFold2MaskedMsaLoss
from ..models.boltzgen.objective import BoltzGenLoss
from ..models.grnade.objective import GrnadeLoss
from ..models.nampnn.objective import NaMpnnLoss
from ..models.nampnn.processor import NaMpnnProcessor
from ..models.odesign.objective import ODesignLoss
from ..models.oinvfold.objective import OInvFoldLoss
from ..models.registry import MODELS, ModelRegistration
from ..modules.backend import Backend
from ..modules.diffusion.loss import edm_denoising_loss
from ..modules.mpnn import MpnnSequenceLoss
from ..processor import Processor
from . import collate
from .config import RunnerConfig
from .dataset import SampledStructureDataset, StructureDataset, collate_with_context
from .sampling import WeightedEpochSampler, read_sampling_index

Collate = Callable[[list[Any]], dict[str, Any]]
Criterion = Callable[[Any, Any], Tensor]


class _TrainingSettings(NamedTuple):
    collate: Collate
    objective: str
    target_source: Literal["structure", "objective"] = "structure"
    find_unused_parameters: bool = False
    forward: Mapping[str, Any] | None = None


def _require_output_classes(trained: Tensor, tokens: Tensor, mask: Tensor) -> None:
    r"""Reject known scored classes outside the model's represented output vocabulary."""
    if isinstance(tokens, NestedTensor):
        tokens = tokens.concat
    if isinstance(mask, NestedTensor):
        mask = mask.concat
    selected = torch.where(mask, tokens, 0)
    valid = ((selected >= 0) & (selected < trained.numel())).all()
    represented = (~mask | trained[selected.clamp(0, trained.numel() - 1)]).all()
    if torch.compiler.is_compiling():
        torch._assert_async(valid & represented, "scored truth classes must be represented by the model")
    elif not bool(valid & represented):
        fail("model_capacity", RequestRef("target", "a known class outside the model output vocabulary"))


def _criterion(objective: str, model: Model, processor: Processor) -> Criterion:
    if objective == "proteinmpnn":
        trained_sequence_loss = MpnnSequenceLoss()

        def trained_class_criterion(prediction: Any, target: Any) -> Tensor:
            # The loss runs over the classes the checkpoint was trained on, so its label smoothing spreads over them
            # only: logits are sliced to those classes and each target class becomes its position among them. A
            # residue whose class the model cannot output (U, O, ...) is not scored.
            features, truth = target["input"], target["target"]
            logits, classes = prediction.logits, truth.tokens
            mask = features.residue_mask.bool() & features.design_mask.bool() & truth.mask
            if isinstance(logits, NestedTensor):
                logits, classes, mask = logits.concat, classes.concat, mask.concat
            classes = torch.where(mask, classes, 0)
            trained = cast(Tensor, model.trained_classes)
            _require_output_classes(trained, classes, mask)
            position = torch.cumsum(trained, dim=0) - 1
            return trained_sequence_loss(logits[..., trained], torch.where(mask, position[classes], 0), mask)

        return trained_class_criterion
    if objective == "ligandmpnn":
        sequence_loss = MpnnSequenceLoss()

        def mpnn_criterion(prediction: Any, target: Any) -> Tensor:
            return sequence_loss(
                prediction.logits,
                target["target"].tokens,
                target["input"].residue_mask.bool() & target["input"].design_mask.bool() & target["target"].mask,
            )

        return mpnn_criterion
    if objective == "nampnn":
        config = cast(NaMpnnProcessor, processor).config
        nampnn_loss = NaMpnnLoss(shared_nucleic_tokens=config.shared_nucleic_tokens)

        def nampnn_criterion(prediction: Any, target: Any) -> Tensor:
            features, truth = target["input"], target["target"]
            return nampnn_loss(
                prediction,
                truth,
                polymer_type_index=features.polymer_type_index,
                mask=features.residue_mask.bool() & features.design_mask.bool(),
            )

        return nampnn_criterion
    if objective == "grnade":
        grnade_loss = GrnadeLoss()

        def grnade_criterion(prediction: Any, target: Any) -> Tensor:
            return grnade_loss(prediction, target["target"], mask=target["input"].design_mask)

        return grnade_criterion
    if objective == "oinvfold":
        oinvfold_loss = OInvFoldLoss()

        def oinvfold_criterion(prediction: Any, target: Any) -> Tensor:
            return oinvfold_loss(prediction, target["target"], mask=target["input"].design_mask)

        return oinvfold_criterion
    if objective == "alphafold2":
        return AlphaFold2MaskedMsaLoss()
    if objective == "denoising":
        return partial(edm_denoising_loss, sigma_data=float(model.config.denoiser.sigma_data), backend=model.backend)
    if objective == "boltzgen_ifold":
        boltzgen_ifold_loss = BoltzGenLoss()

        def boltzgen_ifold_criterion(prediction: Any, target: Any) -> Tensor:
            truth = target["target"]
            tokens = torch.where(truth.mask, truth.tokens, 0)
            residue_type = torch.nn.functional.one_hot(tokens, prediction.logits.shape[-1]).to(prediction.logits)
            return boltzgen_ifold_loss(prediction, target["input"], residue_type)

        return boltzgen_ifold_criterion
    if objective == "odesign":
        odesign_loss_module = ODesignLoss(
            sigma_data=float(model.config.denoiser.sigma_data),
            num_distogram_bins=int(model.config.pairwise.num_distogram_bins),
        )

        def odesign_criterion(prediction: Any, target: Any) -> Tensor:
            return odesign_loss_module(prediction, target["input"], target["target"])

        return odesign_criterion
    raise RuntimeError(f"unknown Runner objective {objective!r}")


_ALPHAFOLD2_TRAINING = _TrainingSettings(
    collate.collate_alphafold2,
    "alphafold2",
    target_source="objective",
    find_unused_parameters=True,
)
_AF3_TRAINING = _TrainingSettings(
    collate.collate_alphafold3_denoising,
    "denoising",
    find_unused_parameters=True,
    forward={"compute_auxiliary_heads": False},
)
_PROTENIX_TRAINING = _TrainingSettings(
    collate.collate_protenix_denoising,
    "denoising",
    find_unused_parameters=True,
    forward={"compute_auxiliary_heads": False},
)
_BOLTZ_TRAINING = _TrainingSettings(
    collate.collate_denoising,
    "denoising",
    find_unused_parameters=True,
)
_BOLTZGEN_TRAINING = _TrainingSettings(
    collate.collate_boltzgen_denoising,
    "denoising",
    find_unused_parameters=True,
)
_ODESIGN_TRAINING = _TrainingSettings(
    collate.collate_odesign,
    "odesign",
)

_TRAINING_SETTINGS = {
    "proteinmpnn": _TrainingSettings(collate.collate_sequence, "proteinmpnn"),
    "nampnn": _TrainingSettings(collate.collate_sequence, "nampnn"),
    "ligandmpnn": _TrainingSettings(collate.collate_sequence, "ligandmpnn"),
    "grnade": _TrainingSettings(collate.collate_grnade, "grnade"),
    "grnade_v0_3_2": _TrainingSettings(collate.collate_grnade, "grnade"),
    "oinvfold": _TrainingSettings(collate.collate_oinvfold, "oinvfold"),
    "alphafold2": _ALPHAFOLD2_TRAINING,
    "alphafold2_initial_guess": _ALPHAFOLD2_TRAINING,
    "alphafold2_multimer": _ALPHAFOLD2_TRAINING,
    "alphafold3": _AF3_TRAINING,
    "protenix": _PROTENIX_TRAINING,
    "protenix_template": _PROTENIX_TRAINING,
    "protenix_structure_condition": _PROTENIX_TRAINING,
    "protenix_constraint": _PROTENIX_TRAINING,
    "boltz1": _BOLTZ_TRAINING,
    "boltz2": _BOLTZ_TRAINING,
    "boltzgen": _BOLTZGEN_TRAINING,
    "boltzgen_miniformer": _BOLTZGEN_TRAINING,
    "boltzgen_ifold": _TrainingSettings(
        collate.collate_boltzgen_ifold,
        "boltzgen_ifold",
    ),
    "odesign": _ODESIGN_TRAINING,
    "odesign_ligand": _ODESIGN_TRAINING,
}


def _prepare(
    processor: Any,
    structure: Structure,
    *,
    target_source: Literal["structure", "objective"],
    target: Structure | None = None,
    **options: Any,
) -> tuple[Any, Any]:
    r"""Prepare one example using the target source declared by its scientific objective."""
    if target_source == "objective":
        if target is not None:
            raise ValueError(
                "this objective derives supervision from its input; an independent structure target is invalid"
            )
        prepared, truth = processor.prepare(structure, **options)
    else:
        prepared, truth = processor.prepare(structure, target=structure if target is None else target, **options)
    return prepared.features, truth


def _training_settings(architecture: str) -> _TrainingSettings:
    try:
        return _TRAINING_SETTINGS[architecture]
    except KeyError:
        raise ValueError(f"architecture {architecture!r} has no built-in Runner objective") from None


class Runner(dl.Runner):
    r"""
    Train and evaluate one registered architecture from a single config.

    ProteinMPNN, LigandMPNN, NA-MPNN, AlphaFold 3, and Protenix collate ragged
    batches without padding. Every other objective reads an Input whose padded
    layout is the family's own, which no general rule concatenates, so it steps
    one example at a time and ``dataloader.batch_size`` must be 1; raise
    ``accum_steps`` for a larger effective batch.

    Every split uses the same ``processor.prepare`` sample interface, returning prepared
    input and a separate typed target. The objective declares whether truth comes from
    an independent structure (the source by default) or from its own input-corruption
    protocol. The Runner adds its automatic cropper only to training splits.

    Runner checkpoints are training artifacts: ``.pth`` files retain model or
    training state, while a native inference repository also records its Processor.
    Export the trained Model and retained Processor as a native repository before
    loading a prediction Pipeline:

    ```python
    from ofoldx import pipeline, save_model

    save_model("./trained-model", model=runner.unwrap(runner.model), processor=runner.processor)
    predictor = pipeline("design", model="./trained-model")
    result = predictor(structure)
    ```

    The task and scientific input follow the trained architecture; this example
    uses a sequence-design Model.

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
        MODELS.for_pair(self.model, self.processor)
        self.model.set_backend(cast(Backend, config.backend))
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.criterion = _criterion(training.objective, self.model, self.processor)
        forward = dict(training.forward or ())
        forward.update(config.forward)
        self.collate_fn = partial(
            collate_with_context,
            collate_fn=training.collate,
            forward=forward,
        )
        self._training = training
        self.cropper = cropper
        self.model.train()
        self.datasets.update(self.build_datasets())

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

    def _execution_data(self, data: Any) -> Any:
        r"""Materialize teacher history without changing the prepared features.

        Generated positions use complete, known target tokens; fixed positions
        retain their input conditioning. Missing generated truth is an error,
        and known classes outside the model's output vocabulary are rejected
        before execution. This keeps the loss domain and its normalizer aligned.
        Training and evaluation use the same construction.
        """
        objective = self._training.objective
        if objective not in {"proteinmpnn", "ligandmpnn", "nampnn", "grnade", "boltzgen_ifold"}:
            return data
        data = self.to_device(data)
        inputs = data["input"]
        arguments = inputs if isinstance(inputs, Mapping) and "input" in inputs else None
        features = arguments["input"] if arguments is not None else inputs
        truth = data["target"]["target"]
        if objective == "boltzgen_ifold":
            required = features.token_mask.bool() & features.token_resolved_mask.bool() & features.design_mask.bool()
            if bool((required & ~truth.mask).any()):
                raise ValueError("BoltzGen teacher forcing requires known truth at every generated token")
            tokens = torch.where(required, truth.tokens, features.native_token_index)
            residue_type = torch.nn.functional.one_hot(tokens, len(TOKENS)).to(features.residue_type_features)
            values = {} if arguments is None else dict(arguments)
            values.update(input=features, target_residue_type=residue_type)
            return {**data, "input": values}
        required = features.design_mask.bool()
        if hasattr(features, "residue_mask"):
            required = required & features.residue_mask.bool()
        if bool((required & ~truth.mask).any()):
            raise ValueError("teacher forcing requires known truth throughout the generated sequence history")
        model = self.unwrap(self.model)
        _require_output_classes(model.trained_classes, truth.tokens, required)
        execution = features.clone(recurse=False)
        execution.sequence_tokens = torch.where(required, truth.tokens, features.sequence_tokens)
        values = execution if arguments is None else {**arguments, "input": execution}
        return {**data, "input": values}

    def train_step(self, data: Any) -> tuple[Any, Tensor | None]:
        return super().train_step(self._execution_data(data))

    def evaluate_step(self, data: Any) -> tuple[Any, Tensor | None]:
        return super().evaluate_step(self._execution_data(data))

    def infer_step(self, data: Any) -> list[Any]:
        r"""
        State that the Runner does not predict.

        A forward call means something different in each family --- logits here, one
        denoising step at a sampled noise level there --- so no single step yields a
        prediction. The Pipeline that owns the family does.
        """

        raise NotImplementedError(
            "the Runner trains and evaluates; export a native repository with "
            "save_model(directory, model=runner.unwrap(runner.model), processor=runner.processor), "
            "then load it with ofoldx.pipeline(...) to predict"
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
            dataset.num_epoch_samples,
            replacement=dataset.replacement,
            seed=dataset.sampling_seed,
            num_replicas=num_replicas,
            rank=rank,
        )

    def build_datasets(self) -> dict[str, Any]:
        processor = partial(_prepare, self.processor, target_source=self._training.target_source)

        datasets: dict[str, Any] = {}
        for split, split_config in self.config.data.items():
            source = split_config.source
            if source is None:
                raise ValueError(f"data.{split}.source is required")
            sampling = split_config.sampling
            processor_kwargs = dict(split_config.processing)
            dataset_options = {"msa_index": split_config.msa_index, "msa_root": split_config.msa_root}
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
                num_epoch_samples=sampling.get("num_epoch_samples"),
                replacement=sampling.get("replacement", True),
                sampling_seed=0 if self.config.seed is None else int(self.config.seed),
                processor_kwargs=processor_kwargs,
                seed=None if self.config.seed is None else int(self.config.seed),
                **dataset_options,
            )
        return datasets
