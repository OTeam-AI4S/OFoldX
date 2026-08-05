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

import random
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, cast

import danling as dl
import torch
from danling.runners.config import normalize_stack_name
from tensordict import is_tensorclass
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing_extensions import Self

from ..model import Model
from ..models.registry import MODELS, ModelRegistration
from ..modules.backend.resolve import Backend
from ..processor import Processor
from . import collate as _collate
from .config import (
    OFOLDX_EDM_RECIPE,
    PROTENIX_BASELINE_RECIPE,
    DatasetMixtureConfig,
    RunnerConfig,
)
from .ema import EMAShadowModel, _unwrap_model

#: The only DanLing stack whose steps this Runner supports.
_SUPPORTED_STACK = "ddp"
_PROTENIX_ARCHITECTURES = frozenset(
    {
        "protenix",
        "protenix_template",
        "protenix_structure_condition",
        "protenix_constraint",
    }
)
_PROTENIX_BASELINE_METRIC_KEYS = tuple(
    [f"objective/{name}" for name in ("mse", "smooth_lddt", "bond", "distogram", "plddt", "pde", "resolved", "pae")]
    + [
        f"objective/weighted_{name}"
        for name in ("mse", "smooth_lddt", "bond", "distogram", "plddt", "pde", "resolved", "pae")
    ]
    + ["objective/loss", "objective/confidence_resolution_valid"]
)


def _data_error_log_glob(training_log_file: str) -> str:
    r"""Return the sidecar glob used by dataset workers for one training log."""

    path = Path(training_log_file)
    return str(path.with_name(f"{path.stem}.data-errors.rank-*.worker-*.jsonl"))


def _resolve_objective(
    architecture: str,
    objective_recipe: str = OFOLDX_EDM_RECIPE,
) -> tuple[
    Callable[..., Callable[[Any, Any], torch.Tensor]],
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

    if objective_recipe == PROTENIX_BASELINE_RECIPE:
        if architecture not in _PROTENIX_ARCHITECTURES:
            raise ValueError(
                f"objective recipe {PROTENIX_BASELINE_RECIPE!r} is supported only by Protenix architectures; "
                f"got {architecture!r}"
            )
        from ..models.protenix.objectives import criterion_factory

        return criterion_factory, _collate.collate_protenix_baseline, True, True
    if objective_recipe != OFOLDX_EDM_RECIPE:
        raise ValueError(f"unsupported objective recipe {objective_recipe!r}")

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
    ema: EMAShadowModel | None
    _resolved_architecture: str

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
        self._resolved_architecture = registration.id
        objective_recipe = config.materialize_objective_recipe(registration.id)
        if objective_recipe is None:  # pragma: no cover - resolved registrations always materialize a recipe.
            raise RuntimeError(f"could not materialize objective_recipe for architecture {registration.id!r}")
        objective_config: Any = None
        if objective_recipe == PROTENIX_BASELINE_RECIPE:
            from ..models.protenix.objectives import ProtenixObjectiveConfig

            configured_objective = config.get("objective_config")
            if configured_objective is None:
                objective_config = ProtenixObjectiveConfig()
            elif isinstance(configured_objective, Mapping):
                objective_config = ProtenixObjectiveConfig(**dict(configured_objective))
            else:
                raise ValueError("objective_config must be a mapping when using protenix_baseline_v1")
            # Freeze the complete preset, including zero-weight terms and
            # chemistry/binning constants, into DanLing's config identity and
            # full-checkpoint metadata before ``super().__init__``.
            config.set("objective_config", asdict(objective_config))
            criterion_factory, collate_fn, process_training, find_unused_parameters = _resolve_objective(
                registration.id, objective_recipe
            )
        else:
            if "objective_config" in config:
                raise ValueError("objective_config is supported only by protenix_baseline_v1")
            # Keep the historical single-argument call for local objective
            # overrides while the versioned EDM recipe preserves that behavior.
            criterion_factory, collate_fn, process_training, find_unused_parameters = _resolve_objective(
                registration.id
            )
        diffusion_batch_size = config.get("diffusion_batch_size")
        diffusion_chunk_size = config.get("diffusion_chunk_size")
        if (diffusion_batch_size is not None or diffusion_chunk_size is not None) and (
            registration.id not in _PROTENIX_ARCHITECTURES
        ):
            raise ValueError(
                "diffusion_batch_size and diffusion_chunk_size are supported only by the Protenix "
                "architectures"
            )
        if registration.id in _PROTENIX_ARCHITECTURES:
            if objective_recipe == PROTENIX_BASELINE_RECIPE:
                model_kwargs: dict[str, Any] = {
                    "training_recipe": PROTENIX_BASELINE_RECIPE,
                    "compute_auxiliary_heads": True,
                    "mini_rollout_steps": 20,
                    "train_confidence_only": bool(objective_config.train_confidence_only),
                }
            else:
                # Preserve the historical EDM-only forward contract. The
                # Protenix model itself defaults to this legacy recipe.
                model_kwargs = {"compute_auxiliary_heads": False}
            if diffusion_batch_size is not None:
                model_kwargs["num_samples"] = diffusion_batch_size
            if diffusion_chunk_size is not None:
                model_kwargs["sample_chunk_size"] = diffusion_chunk_size
            collate_fn = partial(
                _collate.collate_with_model_kwargs,
                collate_fn=collate_fn,
                model_kwargs=model_kwargs,
            )
        if config.ddp.get("find_unused_parameters") is None:
            config.ddp.find_unused_parameters = find_unused_parameters
        super().__init__(config)

        self.model, self.processor = self._build(repository, registration)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        ema_decay = config.get("ema_decay")
        if ema_decay is not None:
            self.ema = EMAShadowModel(self.model, ema_decay)
        if objective_recipe == PROTENIX_BASELINE_RECIPE:
            self.criterion = criterion_factory(self.model, self.processor, objective_config=objective_config)
            # Keep the distributed AverageMeters schema identical even when a
            # confidence term is absent on only some ranks. A zero-count
            # registration contributes nothing to averages.
            for metric_name in _PROTENIX_BASELINE_METRIC_KEYS:
                self.meters[metric_name].update(0.0, n=0)
            self._baseline_current_metric_keys: frozenset[str] = frozenset()
        else:
            self.criterion = criterion_factory(self.model, self.processor)
        self.collate_fn = collate_fn
        self._objective_recipe = objective_recipe
        self._process_training = process_training
        self._forward_cluster_indices = registration.id == "protenix"
        # DanLing toggles module mode from its ``mode`` setter (train_epoch/evaluate_epoch/infer set it), but
        # it fixed ``mode=train`` before this model existed, so the first train_epoch's mode=train is a no-op
        # that will not force train mode -- and a repository model loads in eval(). Establish train mode here.
        self.model.train()
        self.datasets.update(self.build_datasets())
        if bool(self.config.get("logging.enabled", False)) and self.is_main_process and self.logger is not None:
            self.logger.info(
                "dataset processing errors will be recorded in sidecars matching %s",
                _data_error_log_glob(self.workspace.log_file),
            )

    def materialize_model(self) -> None:
        r"""Materialize the live model, then initialize EMA from the DDP-broadcast weights."""

        super().materialize_model()
        if self.ema is not None:
            self.ema.copy_from(self.unwrap(self.model))

    def set_backend(self, backend: Backend) -> Self:
        r"""Set one operator backend consistently on the live and EMA models."""

        cast("Model", _unwrap_model(self.model)).set_backend(backend)
        if self.ema is not None:
            self.ema.set_backend(backend)
        return self

    def load_model(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        r"""Load canonical or legacy compile-prefixed state into the live model."""

        if self.model is None:
            raise ValueError("cannot restore model: model is not initialized")
        wrapped_model = self.unwrap(self.model)
        raw_model = _unwrap_model(self.model)
        checkpoint_keys = set(state_dict)
        raw_keys = set(raw_model.state_dict())
        if checkpoint_keys == raw_keys:
            raw_model.load_state_dict(state_dict, *args, **kwargs)
            return

        # Full compile-enabled checkpoints use the wrapper schema. Prefer its
        # exact keys so their DCP FQNs remain stable across this EMA addition.
        if checkpoint_keys == set(wrapped_model.state_dict()):
            wrapped_model.load_state_dict(state_dict, *args, **kwargs)
            return

        # Direct mapping/file pretrained payloads can also originate from a
        # differently nested compile topology. DCP performs FQN matching before
        # this hook and therefore relies on the exact-wrapper branch above.
        normalized = OrderedDict(state_dict.items())
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            normalized._metadata = metadata.copy()  # type: ignore[attr-defined]
        while normalized and all(name.startswith("_orig_mod.") for name in normalized):
            consume_prefix_in_state_dict_if_present(normalized, "_orig_mod.")
            if set(normalized) == raw_keys:
                raw_model.load_state_dict(normalized, *args, **kwargs)
                return

        # Preserve PyTorch's ordinary missing/unexpected-key diagnostics for
        # genuinely incompatible checkpoints.
        raw_model.load_state_dict(state_dict, *args, **kwargs)

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        r"""Restore full state without reinterpreting legacy Protenix EDM checkpoints.

        For a resolved Protenix architecture, full checkpoints written before
        ``objective_recipe`` existed encode the historical ``ofoldx_edm_v1``
        semantics. They may resume under that explicit recipe, but must never
        pass DanLing's semantic-config check for the new baseline merely because
        the old field was absent. Other architectures canonically omit the key,
        so their checkpoints pass through unchanged.
        """

        runner_config = checkpoint.get("runner")
        if (
            self._resolved_architecture in _PROTENIX_ARCHITECTURES
            and isinstance(runner_config, Mapping)
            and "objective_recipe" not in runner_config
        ):
            current_recipe = self.config.get("objective_recipe")
            if current_recipe != OFOLDX_EDM_RECIPE:
                raise ValueError(
                    "cannot load historical full checkpoint without objective_recipe under "
                    f"{current_recipe!r}: missing recipe means legacy {OFOLDX_EDM_RECIPE!r}; "
                    "select the legacy recipe explicitly or start a new experiment"
                )
            normalized_checkpoint = dict(checkpoint)
            normalized_checkpoint["runner"] = {**runner_config, "objective_recipe": OFOLDX_EDM_RECIPE}
            checkpoint = normalized_checkpoint
        super().load_state_dict(checkpoint)

    def load_pretrained(self, checkpoint: Any, *args: Any, **kwargs: Any) -> None:
        r"""Load pretrained live weights and reset the shadow to the same starting point."""

        super().load_pretrained(checkpoint, *args, **kwargs)
        if self.ema is not None:
            self.ema.copy_from(self.unwrap(self.model))

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

    def _build_structure_dataset(
        self,
        split: str,
        split_config: Any,
        processor: Any,
        *,
        ccd_revision: Any,
        ccd_snapshot_path: Any,
        mixture_component: bool = False,
    ) -> Any:
        r"""Build one ordinary or clustered structure dataset."""

        from .cluster import default_cluster_indices_path
        from .dataset import ClusteredStructureDataset, StructureDataset

        reserved_keys = {
            "source",
            "cluster_sampling",
            "seq_to_pdb_index",
            "msa_path",
            "processing_timeout_seconds",
            "max_consecutive_processing_errors",
            "training_log_file",
        }
        if mixture_component:
            reserved_keys.update({"sampling_ratio", "datasets", "epoch_size", "replacement", "seed"})
        processor_kwargs = {key: value for key, value in split_config.items() if key not in reserved_keys}
        seq_to_pdb_index = split_config.get("seq_to_pdb_index")
        msa_path = split_config.get("msa_path")
        config_get = getattr(self.config, "get", None)
        configured_train_splits = config_get("train_splits") if callable(config_get) else None
        is_train_split = split == "train" if configured_train_splits is None else split in self.train_splits
        processing_timeout_seconds = split_config.get(
            "processing_timeout_seconds", 30.0 if is_train_split else None
        )
        max_consecutive_processing_errors = split_config.get(
            "max_consecutive_processing_errors", 5 if is_train_split else None
        )
        training_log_file = (
            self.workspace.log_file if bool(self.config.get("logging.enabled", False)) else None
        )
        cluster = split_config.cluster_sampling
        if cluster.enabled:
            if not is_train_split:
                raise ValueError(f"cluster sampling is only supported for training splits, not {split!r}")
            if not cluster.cluster_file:
                raise ValueError(f"data.{split}.cluster_sampling.cluster_file is required when enabled")
            if split_config.source is None:
                raise ValueError(f"data.{split}.source is required when cluster sampling is enabled")
            indices_file = getattr(cluster, "indices_file", None)
            if not indices_file:
                workspace = self.config.get("workspace", {})
                workspace_root = workspace.get("root", ".") if isinstance(workspace, Mapping) else workspace.root
                indices_file = default_cluster_indices_path(
                    workspace_root,
                    split_config.source,
                    cluster.cluster_file,
                    contact_cutoff=float(cluster.contact_cutoff),
                    include_all_chain_pairs=bool(cluster.include_all_chain_pairs),
                    exclude_mol_types=tuple(cluster.exclude_mol_types),
                    beta=dict(cluster.beta),
                    alpha=dict(cluster.alpha),
                    ccd_revision=ccd_revision,
                    ccd_snapshot_path=ccd_snapshot_path,
                )
            return ClusteredStructureDataset(
                split_config.source,
                processor,
                cluster_file=cluster.cluster_file,
                cluster_indices_file=indices_file,
                contact_cutoff=float(cluster.contact_cutoff),
                include_all_chain_pairs=bool(cluster.include_all_chain_pairs),
                exclude_mol_types=tuple(cluster.exclude_mol_types),
                beta=dict(cluster.beta),
                alpha=dict(cluster.alpha),
                seq_to_pdb_index=seq_to_pdb_index,
                msa_path=msa_path,
                processor_kwargs=processor_kwargs,
                processing_timeout_seconds=processing_timeout_seconds,
                max_consecutive_processing_errors=max_consecutive_processing_errors,
                training_log_file=training_log_file,
                forward_indices=self._forward_cluster_indices,
                ccd_revision=ccd_revision,
                ccd_snapshot_path=ccd_snapshot_path,
            )
        return StructureDataset(
            split_config.source,
            processor,
            seq_to_pdb_index=seq_to_pdb_index,
            msa_path=msa_path,
            processor_kwargs=processor_kwargs,
            processing_timeout_seconds=processing_timeout_seconds,
            max_consecutive_processing_errors=max_consecutive_processing_errors,
            training_log_file=training_log_file,
            ccd_revision=ccd_revision,
            ccd_snapshot_path=ccd_snapshot_path,
        )

    def build_datasets(self) -> dict[str, Any]:
        r"""
        Build structure datasets for the configured splits. Every split uses the architecture's training
        lowering: sequence families call the Processor, while diffusion families call ``process_training``
        to produce the separate ``(input, target)`` required by their objective.
        """

        from .multidataset import MixtureStructureDataset

        datasets: dict[str, Any] = {}
        processor_config = getattr(self.processor, "config", None)
        if isinstance(processor_config, Mapping):
            ccd_revision = processor_config.get("ccd_revision")
            ccd_snapshot_path = processor_config.get("ccd_snapshot_path")
        else:
            ccd_revision = getattr(processor_config, "ccd_revision", None)
            ccd_snapshot_path = getattr(processor_config, "ccd_snapshot_path", None)
        processor: Any = self.processor
        if self._process_training:
            processor = processor.process_training
        for split, split_config in self.config.data.items():
            if isinstance(split_config, DatasetMixtureConfig):
                configured_train_splits = self.config.get("train_splits")
                is_train_split = split == "train" if configured_train_splits is None else split in self.train_splits
                if not is_train_split:
                    raise ValueError(f"dataset mixtures are only supported for training splits, not {split!r}")
                child_datasets = {
                    name: Runner._build_structure_dataset(
                        self,
                        split,
                        split_config.datasets[name],
                        processor,
                        ccd_revision=ccd_revision,
                        ccd_snapshot_path=ccd_snapshot_path,
                        mixture_component=True,
                    )
                    for name in sorted(split_config.datasets)
                }
                sampling_ratios = {
                    name: float(split_config.datasets[name].sampling_ratio) for name in sorted(split_config.datasets)
                }
                datasets[split] = MixtureStructureDataset(child_datasets, sampling_ratios)
            else:
                datasets[split] = Runner._build_structure_dataset(
                    self,
                    split,
                    split_config,
                    processor,
                    ccd_revision=ccd_revision,
                    ccd_snapshot_path=ccd_snapshot_path,
                )
        return datasets

    def build_datasampler(self, dataset: Any, *, split: str, shuffle: bool) -> Any:
        r"""Use row-weighted cluster sampling only for explicitly enabled training splits."""

        from .cluster import ClusterWeightedSampler
        from .dataset import ClusteredStructureDataset
        from .multidataset import MixtureStructureDataset, MixtureWeightedSampler

        if isinstance(dataset, MixtureStructureDataset):
            if split not in self.train_splits:
                raise ValueError(f"dataset mixtures are only supported for training splits, not {split!r}")
            mixture = self.config.data[split]
            epoch_size = len(dataset) if mixture.epoch_size is None else mixture.epoch_size
            seed = self.config.seed if mixture.seed is None else mixture.seed
            num_replicas, rank = 1, 0
            if self.distributed:
                num_replicas, rank = self.world_size, self.rank
                if self.fault_tolerance is not None:
                    num_replicas, rank = self.fault_tolerance.data_parallel_info(num_replicas, rank)
            return MixtureWeightedSampler(
                dataset.sampling_weights,
                int(epoch_size),
                replacement=bool(mixture.replacement),
                seed=0 if seed is None else int(seed),
                num_replicas=num_replicas,
                rank=rank,
            )

        if not isinstance(dataset, ClusteredStructureDataset) or split not in self.train_splits:
            return super().build_datasampler(dataset, split=split, shuffle=shuffle)

        cluster = self.config.data[split].cluster_sampling
        epoch_size = cluster.epoch_size
        if epoch_size is None:
            epoch_size = len(dataset.entries)
        seed = self.config.seed if cluster.seed is None else cluster.seed
        num_replicas, rank = 1, 0
        if self.distributed:
            num_replicas, rank = self.world_size, self.rank
            if self.fault_tolerance is not None:
                num_replicas, rank = self.fault_tolerance.data_parallel_info(num_replicas, rank)
        return ClusterWeightedSampler(
            dataset.entries,
            dataset.weights,
            int(epoch_size),
            replacement=bool(cluster.replacement),
            seed=0 if seed is None else int(seed),
            num_replicas=num_replicas,
            rank=rank,
        )

    # -- steps ---------------------------------------------------------------

    def _sync_baseline_nonfinite_loss(self, loss: torch.Tensor) -> bool:
        r"""Return a replica-synchronized non-finite decision for one baseline loss."""

        has_nonfinite = not bool(torch.isfinite(loss.detach()).all().item())
        if not (
            self.distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return has_nonfinite
        payload = torch.tensor(float(has_nonfinite), device=self.all_reduce_device())
        self.all_reduce(payload, op=torch.distributed.ReduceOp.MAX)
        return bool(payload.item() > 0)

    def _record_baseline_objective_metrics(self) -> None:
        r"""Publish the current finite criterion metrics through DanLing meters."""

        latest_metrics = getattr(self.criterion, "latest_metrics", {})
        if not isinstance(latest_metrics, Mapping):
            self._baseline_current_metric_keys = frozenset()
            return

        published: set[str] = set()
        for metric_name in _PROTENIX_BASELINE_METRIC_KEYS:
            metric = latest_metrics.get(metric_name)
            if not isinstance(metric, torch.Tensor) or metric.numel() != 1:
                continue
            metric = metric.detach()
            if not bool(torch.isfinite(metric).all().item()):
                continue
            self.meters[metric_name].update(metric)
            published.add(metric_name)
        self._baseline_current_metric_keys = frozenset(published)

    def _sync_preserved_baseline_gradients(
        self,
        gradients: list[tuple[torch.nn.Parameter, torch.Tensor | None]],
    ) -> None:
        r"""Average accumulated gradients after an invalid DDP boundary micro-batch."""

        divisor = self._loss_normalizer_sync_divisor()
        for parameter, gradient in gradients:
            present = torch.tensor(float(gradient is not None), device=self.all_reduce_device())
            self.all_reduce(present, op=torch.distributed.ReduceOp.MAX)
            if not bool(present.item() > 0):
                parameter.grad = None
                continue
            synchronized = torch.zeros_like(parameter) if gradient is None else gradient
            self.all_reduce(synchronized, op=torch.distributed.ReduceOp.SUM)
            synchronized.div_(divisor)
            parameter.grad = synchronized

    def _baseline_zero_backward_preserving_gradients(
        self,
        loss: torch.Tensor,
        *,
        synchronize_accumulated_gradients: bool,
    ) -> None:
        r"""Finish the DDP reducer with a zero graph, then restore accumulated gradients.

        A standalone leaf-zero backward matches the upstream numerical intent
        but leaves DDP's reducer prepared for a backward that never reaches its
        hooks. Connecting a zero to the current loss graph completes those
        hooks. Restoring every pre-backward gradient afterward prevents newly
        materialized zero gradients from changing weight-decay or optimizer
        state and preserves earlier micro-batches in the accumulation window.
        """

        gradients = [
            (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in self.iter_optimizer_parameters()
        ]
        independent_zero = loss.detach().new_zeros((), requires_grad=True)
        if loss.requires_grad:
            finite_loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
            zero_loss = independent_zero + finite_loss.sum() * 0.0
        else:  # pragma: no cover - the baseline criterion always returns a differentiable scalar.
            zero_loss = independent_zero
        try:
            self.backward(zero_loss)
        finally:
            for parameter, gradient in gradients:
                parameter.grad = gradient
        if synchronize_accumulated_gradients:
            self._sync_preserved_baseline_gradients(gradients)

    def get_step_result(self) -> Any:
        r"""Hide baseline objective values absent from the current micro-batch."""

        result = super().get_step_result()
        if self._objective_recipe != PROTENIX_BASELINE_RECIPE:
            return result
        current_keys = getattr(self, "_baseline_current_metric_keys", frozenset())
        for metric_name in _PROTENIX_BASELINE_METRIC_KEYS:
            if metric_name not in current_keys:
                result.pop(metric_name, None)
        return result

    def get_epoch_result(self) -> Any:
        r"""Drop globally absent baseline metrics instead of reporting NaN."""

        result = super().get_epoch_result()
        if self._objective_recipe != PROTENIX_BASELINE_RECIPE:
            return result
        for metric_name in _PROTENIX_BASELINE_METRIC_KEYS:
            value = result.get(metric_name)
            if isinstance(value, torch.Tensor):
                is_finite = bool(torch.isfinite(value).all().item())
            else:
                try:
                    is_finite = bool(torch.isfinite(torch.tensor(float(value))).item())
                except (TypeError, ValueError):
                    is_finite = False
            if not is_finite:
                result.pop(metric_name, None)
        return result

    def train_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        r"""Run one recipe-specific training micro-step.

        EDM delegates unchanged to DanLing. Baseline injects the synchronized
        ``global_step`` recycle seed and adds a pre-backward cross-rank loss
        gate while retaining DanLing's accumulation, step, scheduler, and EMA
        lifecycle.
        """

        if self._objective_recipe != PROTENIX_BASELINE_RECIPE:
            return super().train_step(data)
        if not isinstance(data, Mapping):
            raise TypeError("the Protenix baseline objective requires a mapping batch")
        inputs = data.get("input")
        if not isinstance(inputs, Mapping):
            raise TypeError("the Protenix baseline objective requires a mapping under batch['input']")
        if "current_step" in inputs:
            raise ValueError("current_step is Runner-controlled and must not be supplied by the data loader")
        seeded_data = {**data, "input": {**inputs, "current_step": self.train_state.global_step}}
        seeded_data = self.to_device(seeded_data)
        with self.train_context():
            model_inputs = seeded_data["input"]
            target = seeded_data.get("target")
            if self.model is None:
                raise ValueError("cannot run train_step: model is not initialized")
            prediction = self.model(**model_inputs)
            loss = self.criterion(prediction, target) if self.criterion is not None else None
            if loss is None:
                raise ValueError("cannot run train_step: criterion did not produce a loss")

            if self._sync_baseline_nonfinite_loss(loss):
                self._baseline_current_metric_keys = frozenset()
                synchronize_accumulated_gradients = bool(self._train_no_sync_targets()) and not (
                    self._should_train_no_sync()
                )
                self._baseline_zero_backward_preserving_gradients(
                    loss,
                    synchronize_accumulated_gradients=synchronize_accumulated_gradients,
                )
                returned_loss = loss.detach().new_zeros(())
            else:
                self._record_baseline_objective_metrics()
                if self.metrics is not None and prediction is not None and target is not None:
                    self.metrics.update(prediction, target)
                self.backward(loss)
                returned_loss = loss
            self.step()
        return prediction, returned_loss

    @staticmethod
    def _draw_baseline_mc_dropout() -> bool:
        r"""Draw the released per-structure Monte Carlo dropout decision."""

        return random.random() < 0.4

    def evaluate_step(self, data: Any) -> tuple[Any, torch.Tensor | None]:
        r"""Run recipe-specific labeled evaluation without mutating training state.

        EDM delegates unchanged to DanLing. Baseline evaluation samples five
        reverse-diffusion structures with the model's 200-step/full-recycling
        defaults, canonicalizes every sample and confidence head prediction-side,
        and carries the shared distogram's contact probabilities in both shared
        and per-sample canonical order before invoking the evaluation-only
        objective. EMA is selected only when configured, and training-free
        guidance is explicitly disabled.
        """

        if self._objective_recipe != PROTENIX_BASELINE_RECIPE:
            return super().evaluate_step(data)
        if not isinstance(data, Mapping):
            raise TypeError("the Protenix baseline evaluation requires a mapping batch")
        data = self.to_device(data)
        inputs = data.get("input")
        if not isinstance(inputs, Mapping):
            raise TypeError("the Protenix baseline evaluation requires a mapping under batch['input']")
        model_input = inputs.get("input")
        structure_target = inputs.get("target")
        if model_input is None or structure_target is None:
            raise ValueError("the Protenix baseline evaluation requires collated input and target structures")

        ema = getattr(self, "ema", None)
        if ema is not None:
            evaluation_model: Any = ema
            # Runner mode transitions primarily own the live model. Keep the
            # independently held shadow deterministic even for direct calls.
            evaluation_model.eval()
        else:
            if self.model is None:
                raise ValueError("cannot run evaluate_step: model is not initialized")
            evaluation_model = _unwrap_model(self.model)
        sample = getattr(evaluation_model, "sample", None)
        if not callable(sample):
            raise AttributeError(f"{type(evaluation_model).__name__} does not provide a callable `sample` method")
        evaluate = getattr(self.criterion, "evaluate", None)
        if not callable(evaluate):
            raise AttributeError("the Protenix baseline criterion does not provide a callable `evaluate` method")

        from ..models.protenix.model import (
            ProtenixEvaluationOutput,
            ProtenixRawOutput,
            protenix_contact_probability,
        )
        from ..models.protenix.permutation import match_protenix_prediction

        mc_dropout = self._draw_baseline_mc_dropout()
        with self.infer_context():
            raw_prediction = sample(
                model_input,
                evaluation_recipe=PROTENIX_BASELINE_RECIPE,
                mc_dropout=mc_dropout,
                mc_dropout_rate=0.4,
                tfg_guidance=None,
            )
            if not isinstance(raw_prediction, ProtenixRawOutput):
                raise TypeError(
                    "Protenix baseline sample must return ProtenixRawOutput, got "
                    f"{type(raw_prediction).__name__}"
                )
            permutation = match_protenix_prediction(
                raw_prediction.coordinates,
                structure_target,
                model_input,
            )
            heads = permutation.gather_heads(
                plddt_logits=raw_prediction.plddt_logits,
                resolved_logits=raw_prediction.resolved_logits,
                pae_logits=raw_prediction.pae_logits,
                pde_logits=raw_prediction.pde_logits,
            )
            contact_probs = protenix_contact_probability(raw_prediction.distogram_logits)
            prediction = ProtenixEvaluationOutput(  # type: ignore[call-arg]
                coordinates=permutation.coordinates,
                distogram_logits=raw_prediction.distogram_logits,
                **heads,
                contact_probs=contact_probs,
                per_sample_contact_probs=permutation.gather_shared_token_pair_axes(contact_probs),
                batch_size=raw_prediction.batch_size,
            )
            objective_target = {"input": model_input, "target": structure_target}
            loss = evaluate(prediction, objective_target)

        self._record_baseline_objective_metrics()
        if self.metrics is not None:
            self.metrics.update(prediction, objective_target)
        return prediction, loss

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
