r"""
Configuration values for the OFoldX runner.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import chanfig
import danling as dl
from danling.runners.config import DdpConfig as DanlingDdpConfig


_INTERPOLATION_PATTERN = re.compile(r"\$\{[^{}]+\}")
_DATASET_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_MIXTURE_KEYS = frozenset({"datasets", "epoch_size", "replacement", "seed"})
_COMPONENT_CONTROL_KEYS = frozenset({"epoch_size", "replacement", "seed"})
_PROTENIX_MODEL_NAMES = frozenset(
    {
        "protenix",
        "protenix_template",
        "protenix_structure_condition",
        "protenix_constraint",
    }
)
OBJECTIVE_RECIPES = frozenset({"protenix_baseline_v1", "ofoldx_edm_v1"})
PROTENIX_BASELINE_RECIPE = "protenix_baseline_v1"
OFOLDX_EDM_RECIPE = "ofoldx_edm_v1"
_PROTENIX_BASELINE_DIFFUSION_BATCH_SIZE = 48
_PROTENIX_BASELINE_DIFFUSION_CHUNK_SIZE = 4
_CLUSTER_SAMPLING_KEYS = frozenset(
    {
        "enabled",
        "cluster_file",
        "indices_file",
        "epoch_size",
        "replacement",
        "contact_cutoff",
        "include_all_chain_pairs",
        "exclude_mol_types",
        "seed",
        "beta",
        "alpha",
    }
)


def _contains_interpolation(value: Any) -> bool:
    return isinstance(value, str) and _INTERPOLATION_PATTERN.search(value) is not None


def _normalize_path_value(value: Any) -> Any:
    r"""Convert text ``PathLike`` values without coercing other scalar types."""

    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _resolved_value(value: Any) -> Any:
    r"""Return a chanfig interpolation target without its ``Variable`` wrapper."""

    return value.value if isinstance(value, chanfig.Variable) else value


def _raw_matches(value: Any, expected_type: type | tuple[type, ...], *, allow_none: bool = False) -> bool:
    return (allow_none and value is None) or isinstance(value, expected_type) or _contains_interpolation(value)


def _is_positive_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _set_plain_value(config: chanfig.Config, name: str, value: Any) -> None:
    r"""Detach a normalized path from an interpolated ``Variable`` safely."""

    dict.__setitem__(config, name, value)
    object.__setattr__(config, name, value)


def _normalize_optional_ema_decay(values: dict[Any, Any]) -> None:
    r"""Validate an explicitly configured EMA decay without adding a default."""

    if "ema_decay" not in values:
        return
    value = values["ema_decay"]
    if value is None:
        # Absence is the canonical disabled state. In particular, do not leave
        # a ``None`` key behind because config contents determine run identity.
        del values["ema_decay"]
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("ema_decay must be a finite number strictly between 0 and 1, or None")
    try:
        decay = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("ema_decay must be a finite number strictly between 0 and 1, or None") from error
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("ema_decay must be a finite number strictly between 0 and 1, or None")
    values["ema_decay"] = decay


def _normalize_optional_positive_int(values: dict[Any, Any], key: str) -> None:
    r"""Validate an explicitly configured positive integer without adding a default."""

    if key not in values:
        return
    value = values[key]
    if value is None:
        # Omitted and explicit-null configurations intentionally have identical
        # run identities.
        del values[key]
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer or None")
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer or None")


class ModelConfig(chanfig.Config):
    r"""
    Select and configure the model built by a runner.

    Additional configuration entries override architecture settings when
    ``pretrained`` is absent.

    Attributes:
        name: Registered architecture ID.
        pretrained: Optional local model repository or Hugging Face Hub repository ID.
        revision: Optional Hub revision (branch, tag, or commit) to resolve ``pretrained`` at. On load the
            Runner records the resolved commit sha back here, so a mutable id (branch/tag) resumes from the
            exact snapshot it was trained on.
    """

    name: str = ""
    pretrained: str | None = None
    revision: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = ""
        self.pretrained = None
        self.revision = None
        super().__init__(*args, **kwargs)


class ClusterSamplingConfig(chanfig.Config):
    r"""Protenix-style cluster-weighted sampling for a training CIF split.

    Example:
        ``{"data": {"train": {"source": "train.jsonl", "cluster_sampling":``
        ``{"enabled": True, "cluster_file": "clusters.txt", "epoch_size": 10000}}}}``

    This asymmetric-unit-only mode caches weighted index rows and scans all CIFs
    only when the cache is absent or invalid. Rows map back to full CIF
    records/structures and carry chain/interface metadata that can produce
    distinct Protenix crops. Default structure ingestion removes hydrogens and
    water; configured ion handling and ``exclude_mol_types`` change sampling
    rows only. ``epoch_size`` is the global draw count and DDP pads it upward
    so every rank receives
    ``ceil(epoch_size / world_size)`` rows.
    """

    # These values intentionally use ``Any``: mixture validation is strict,
    # while chanfig's annotation honoring would otherwise coerce invalid
    # interpolated scalars (for example ``1`` to ``True``).
    enabled: Any = False
    cluster_file: Any = None
    indices_file: Any = None
    epoch_size: Any = None
    replacement: Any = True
    contact_cutoff: Any = 5.0
    include_all_chain_pairs: Any = False
    exclude_mol_types: list[str]
    seed: Any = None
    beta: chanfig.Config
    alpha: chanfig.Config

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        input_keys = frozenset(values)
        values["cluster_file"] = _normalize_path_value(values.get("cluster_file"))
        values["indices_file"] = _normalize_path_value(values.get("indices_file"))
        enabled = values.get("enabled", False)
        cluster_file = values.get("cluster_file")
        indices_file = values.get("indices_file")
        replacement = values.get("replacement", True)
        contact_cutoff = values.get("contact_cutoff", 5.0)
        include_all_chain_pairs = values.get("include_all_chain_pairs", False)
        exclude_mol_types = values.get("exclude_mol_types", ["ions"])
        self.enabled = False
        self.cluster_file = None
        self.indices_file = None
        self.epoch_size = None
        self.replacement = True
        self.contact_cutoff = 5.0
        self.include_all_chain_pairs = False
        self.exclude_mol_types = ["ions"]
        self.seed = None
        self.beta = chanfig.Config({"chain": 0.5, "interface": 1.0})
        self.alpha = chanfig.Config({"prot": 3.0, "nuc": 3.0, "ligand": 1.0})
        super().__init__(values)
        # Preserve raw input types that chanfig annotations may otherwise
        # coerce before RunnerConfig validates a mixture.
        self.setattr("_input_keys", input_keys)
        self.setattr("_enabled_raw_valid", _raw_matches(enabled, bool))
        self.setattr("_cluster_file_raw_valid", _raw_matches(cluster_file, str, allow_none=True))
        self.setattr("_indices_file_raw_valid", _raw_matches(indices_file, str, allow_none=True))
        self.setattr("_replacement_raw_valid", _raw_matches(replacement, bool))
        self.setattr("_contact_cutoff_is_bool", isinstance(contact_cutoff, bool))
        self.setattr("_include_all_chain_pairs_raw_valid", _raw_matches(include_all_chain_pairs, bool))
        self.setattr(
            "_exclude_mol_types_is_valid",
            isinstance(exclude_mol_types, (list, tuple))
            and all(isinstance(mol_type, str) for mol_type in exclude_mol_types),
        )


class DatasetConfig(chanfig.Config):
    r"""
    Configure one structure-dataset split.

    Additional configuration entries are forwarded to the processor for each item.

    Attributes:
        source: Directory, glob, manifest, structure file, or ``None``.
        seq_to_pdb_index: Optional JSON mapping from exact protein sequences to
            indexed MSA directory names.
        msa_path: Optional root directory for indexed MSAs. Must be configured
            together with ``seq_to_pdb_index``.
    """

    source: Any = None
    seq_to_pdb_index: Any = None
    msa_path: Any = None
    cluster_sampling: ClusterSamplingConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        cluster_sampling = values.pop("cluster_sampling", None)
        for path_key in ("source", "seq_to_pdb_index", "msa_path"):
            if path_key in values:
                values[path_key] = _normalize_path_value(values[path_key])
        self.source = None
        self.seq_to_pdb_index = None
        self.msa_path = None
        self.cluster_sampling = ClusterSamplingConfig()
        super().__init__(values)
        if cluster_sampling is not None:
            self.cluster_sampling = (
                cluster_sampling
                if isinstance(cluster_sampling, ClusterSamplingConfig)
                else ClusterSamplingConfig(cluster_sampling)
            )


class MixtureComponentConfig(DatasetConfig):
    r"""Configure one named component of a training-dataset mixture."""

    sampling_ratio: Any = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        input_keys = frozenset(values)
        for path_key in ("source", "seq_to_pdb_index", "msa_path"):
            if path_key in values:
                values[path_key] = _normalize_path_value(values[path_key])
        source = values.get("source")
        sampling_ratio = values.get("sampling_ratio", 1.0)
        seq_to_pdb_index = values.get("seq_to_pdb_index")
        msa_path = values.get("msa_path")
        processing_timeout_seconds = values.get("processing_timeout_seconds")
        max_consecutive_processing_errors = values.get("max_consecutive_processing_errors")
        self.sampling_ratio = 1.0
        super().__init__(values)
        # chanfig may coerce an annotated bool to ``1.0`` before post-time
        # validation, so retain this small piece of input-type information.
        self.setattr("_input_keys", input_keys)
        self.setattr("_source_raw_valid", _raw_matches(source, str))
        self.setattr(
            "_sampling_ratio_raw_valid",
            (
                isinstance(sampling_ratio, Real)
                and not isinstance(sampling_ratio, bool)
                or _contains_interpolation(sampling_ratio)
            ),
        )
        self.setattr("_seq_to_pdb_index_raw_valid", _raw_matches(seq_to_pdb_index, str, allow_none=True))
        self.setattr("_msa_path_raw_valid", _raw_matches(msa_path, str, allow_none=True))
        self.setattr(
            "_processing_timeout_seconds_raw_valid",
            _raw_matches(processing_timeout_seconds, Real, allow_none=True)
            and not isinstance(processing_timeout_seconds, bool),
        )
        self.setattr(
            "_max_consecutive_processing_errors_raw_valid",
            _raw_matches(max_consecutive_processing_errors, int, allow_none=True)
            and not isinstance(max_consecutive_processing_errors, bool),
        )


class DatasetMixtureConfig(chanfig.Config):
    r"""Configure weighted sampling across named training datasets."""

    datasets: Mapping[str, MixtureComponentConfig]
    epoch_size: Any = None
    replacement: Any = True
    seed: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        input_keys = frozenset(values)
        datasets = values.pop("datasets", {})
        epoch_size = values.get("epoch_size")
        replacement = values.get("replacement", True)
        seed = values.get("seed")
        self.datasets = {}
        self.epoch_size = None
        self.replacement = True
        self.seed = None
        super().__init__(values)
        self.setattr("_input_keys", input_keys)
        self.setattr(
            "_epoch_size_raw_valid",
            _raw_matches(epoch_size, int, allow_none=True) and not isinstance(epoch_size, bool),
        )
        self.setattr("_replacement_raw_valid", _raw_matches(replacement, bool))
        self.setattr("_seed_raw_valid", _raw_matches(seed, int, allow_none=True) and not isinstance(seed, bool))
        if isinstance(datasets, Mapping):
            # A Config container makes component configs part of chanfig's boot
            # tree (including interpolation). Bypass NestedDict.set here so a
            # malformed dotted name remains intact for the validator to report.
            converted_datasets = chanfig.Config()
            for name, component in datasets.items():
                dict.__setitem__(
                    converted_datasets,
                    name,
                    component if isinstance(component, MixtureComponentConfig) else MixtureComponentConfig(component),
                )
            datasets = converted_datasets
        self.set("datasets", datasets, convert_mapping=False)


class DataConfig(chanfig.Config):
    r"""
    Map split names to [`DatasetConfig`][ofoldx.runners.config.DatasetConfig] values.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        super().__init__()
        for split, spec in values.items():
            if isinstance(spec, (DatasetConfig, DatasetMixtureConfig)):
                converted = spec
            elif isinstance(spec, Mapping) and "datasets" in spec:
                converted = DatasetMixtureConfig(spec)
            else:
                converted = DatasetConfig(spec) if isinstance(spec, Mapping) else DatasetConfig(source=spec)
            self.set(split, converted, convert_mapping=False)


class DdpConfig(DanlingDdpConfig):
    r"""DDP settings whose unused-parameter policy defaults to the selected objective."""

    find_unused_parameters: bool | None = None  # type: ignore[assignment]


class RunnerConfig(dl.RunnerConfig):
    r"""
    Configure OFoldX training and evaluation.

    Danling optimizer, scheduler, dataloader, checkpoint, and scoring options are
    inherited from [`danling.RunnerConfig`][danling.RunnerConfig].

    Attributes:
        model: Model selection and architecture (model-config) overrides.
        processor: Processor-config overrides that are not model settings
            (e.g. NA-MPNN ``shared_nucleic_tokens``).
        data: Dataset configuration keyed by split name.
        gradient_checkpointing: Enable model-native activation checkpointing
            during training. This is independent of training-state checkpoint
            persistence.
        objective_recipe: Versioned training-objective contract. Fresh
            Protenix configurations default to ``protenix_baseline_v1``. Other
            architectures use ``ofoldx_edm_v1`` internally without persisting
            that historical default in the configuration identity.
        diffusion_batch_size: Optional number of independently corrupted
            diffusion samples per input structure.
        diffusion_chunk_size: Optional maximum number of diffusion samples
            evaluated by one denoiser call.
        ema_decay: Optional model-parameter exponential moving-average decay.
            The key is absent when EMA is disabled so legacy configuration
            identities remain unchanged.
    """

    model: ModelConfig
    processor: chanfig.Config
    data: DataConfig
    ddp: DdpConfig = DdpConfig()
    gradient_checkpointing: bool = False
    objective_recipe: str
    diffusion_batch_size: int
    diffusion_chunk_size: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # NestedDict expands dots in mapping keys during construction. Reject
        # invalid raw mixture names before that transformation makes the
        # original spelling unrecoverable. Non-mapping inputs continue down
        # the parent constructor unchanged.
        normalized_args = args
        if len(args) == 1 and isinstance(args[0], Mapping):
            values = dict(args[0])
            self._validate_raw_dataset_names(values)
            self._validate_raw_objective_recipe(values)
            _normalize_optional_ema_decay(values)
            _normalize_optional_positive_int(values, "diffusion_batch_size")
            _normalize_optional_positive_int(values, "diffusion_chunk_size")
            normalized_args = (values,)
        normalized_kwargs = dict(kwargs)
        if normalized_kwargs:
            self._validate_raw_dataset_names(normalized_kwargs)
            self._validate_raw_objective_recipe(normalized_kwargs)
            disable_positional_ema = normalized_kwargs.get("ema_decay", object()) is None
            disabled_positional_diffusion = {
                key
                for key in ("diffusion_batch_size", "diffusion_chunk_size")
                if key in normalized_kwargs and normalized_kwargs[key] is None
            }
            _normalize_optional_ema_decay(normalized_kwargs)
            _normalize_optional_positive_int(normalized_kwargs, "diffusion_batch_size")
            _normalize_optional_positive_int(normalized_kwargs, "diffusion_chunk_size")
            if disable_positional_ema and len(normalized_args) == 1 and isinstance(normalized_args[0], dict):
                normalized_args[0].pop("ema_decay", None)
            if len(normalized_args) == 1 and isinstance(normalized_args[0], dict):
                for key in disabled_positional_diffusion:
                    normalized_args[0].pop(key, None)
        super().__init__(*normalized_args, **normalized_kwargs)

    @staticmethod
    def _validate_raw_objective_recipe(values: Mapping[Any, Any]) -> None:
        if "objective_recipe" not in values:
            return
        recipe = values["objective_recipe"]
        if not isinstance(recipe, str) and not _contains_interpolation(recipe):
            raise ValueError(
                "objective_recipe must be one of " + ", ".join(sorted(OBJECTIVE_RECIPES))
            )

    @staticmethod
    def _validate_raw_dataset_names(values: Mapping[Any, Any]) -> None:
        data = values.get("data")
        if not isinstance(data, Mapping):
            return
        for split, split_config in data.items():
            if not isinstance(split_config, Mapping):
                continue
            datasets = split_config.get("datasets")
            if not isinstance(datasets, Mapping):
                continue
            for name in datasets:
                if not isinstance(name, str) or _DATASET_NAME_PATTERN.fullmatch(name) is None:
                    raise ValueError(
                        f"data.{split}.datasets contains invalid name {name!r}; "
                        "expected [A-Za-z][A-Za-z0-9_-]*"
                    )

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "model" not in self:
            self.model = ModelConfig()
        elif not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(self.model)
        if "processor" not in self:
            self.processor = chanfig.Config()
        elif not isinstance(self.processor, chanfig.Config):
            self.processor = chanfig.Config(self.processor)
        if "data" not in self:
            self.data = DataConfig()
        elif not isinstance(self.data, DataConfig):
            self.data = DataConfig(self.data)
        self.materialize_objective_recipe()
        self._validate_diffusion_architecture()

    def materialize_objective_recipe(self, architecture: str | None = None) -> str | None:
        r"""Resolve and canonicalize the versioned objective before identity is frozen.

        A pretrained repository may omit ``model.name`` until its metadata is
        resolved. In that one case this method intentionally defers the default;
        :class:`Runner` calls it again with the resolved registration before
        DanLing derives the configuration hash. Only Protenix recipes are part
        of persisted identity; other architectures use the historical EDM
        objective as a runtime fallback with no canonical config key.
        """

        model_name = architecture if architecture is not None else _resolved_value(self.model.name)
        if _contains_interpolation(model_name):
            return None
        if model_name is not None and not isinstance(model_name, str):
            raise ValueError("model.name must resolve to a string before selecting objective_recipe")

        if "objective_recipe" in self:
            recipe = _resolved_value(self.get("objective_recipe"))
            if _contains_interpolation(recipe):
                return None
            if not isinstance(recipe, str) or recipe not in OBJECTIVE_RECIPES:
                raise ValueError(
                    "objective_recipe must be one of " + ", ".join(sorted(OBJECTIVE_RECIPES))
                )
            _set_plain_value(self, "objective_recipe", recipe)
        else:
            recipe = None

        # Repository metadata, rather than a possibly absent model.name, owns
        # the final recipe decision. Preserve an explicit recipe only long
        # enough for Runner to call this method with the resolved registration.
        if not model_name:
            return None

        if recipe == PROTENIX_BASELINE_RECIPE and model_name and model_name not in _PROTENIX_MODEL_NAMES:
            raise ValueError(
                f"objective_recipe {PROTENIX_BASELINE_RECIPE!r} is supported only by Protenix architectures; "
                f"got {model_name!r}"
            )

        if model_name not in _PROTENIX_MODEL_NAMES:
            if "objective_recipe" in self:
                del self["objective_recipe"]
            return OFOLDX_EDM_RECIPE

        if recipe is None:
            recipe = PROTENIX_BASELINE_RECIPE
            _set_plain_value(self, "objective_recipe", recipe)

        if recipe == PROTENIX_BASELINE_RECIPE:
            if "diffusion_batch_size" not in self:
                _set_plain_value(self, "diffusion_batch_size", _PROTENIX_BASELINE_DIFFUSION_BATCH_SIZE)
            if "diffusion_chunk_size" not in self:
                _set_plain_value(self, "diffusion_chunk_size", _PROTENIX_BASELINE_DIFFUSION_CHUNK_SIZE)
        return recipe

    def _validate_diffusion_architecture(self) -> None:
        r"""Reject diffusion batching for a known fresh non-Protenix model."""

        controls_configured = any(key in self for key in ("diffusion_batch_size", "diffusion_chunk_size"))
        model_name = _resolved_value(self.model.name)
        if _contains_interpolation(model_name):
            return
        if (
            controls_configured
            and model_name
            and not self.model.pretrained
            and model_name not in _PROTENIX_MODEL_NAMES
        ):
            raise ValueError(
                "diffusion_batch_size and diffusion_chunk_size are supported only by the Protenix architectures"
            )

    def post(self) -> None:
        r"""
        Finalize and validate the runner configuration.

        The model is selected either by ``model.pretrained`` (a repository whose snapshot supplies both
        the model and its processor) or by ``model.name`` (a registered architecture built fresh). When
        a repository is given, ``model.name`` is optional and, if present, is asserted against the loaded
        architecture at build time.

        Raises:
            ValueError: If neither ``model.pretrained`` nor ``model.name`` is set.
        """

        super().post()
        self._validate_diffusion_architecture()
        if not self.model.pretrained and not self.model.name:
            raise ValueError(
                "select a model with `model.pretrained` (a repository) or `model.name` "
                "(a registered architecture id)"
            )
        # ``Config.boot`` invokes ``post`` before interpolation and framework
        # type validation. Defer mixture semantics until ``boot`` finishes;
        # direct callers of ``post`` still get the historical eager checks.
        if not self.getattr("_booting_mixtures", False):
            self._validate_dataset_mixtures()

    def boot(self) -> RunnerConfig:
        r"""Finalize chanfig first, then validate interpolation-aware mixture semantics."""

        self.setattr("_booting_mixtures", True)
        try:
            super().boot()
            self.materialize_objective_recipe()
            self._validate_dataset_mixtures()
        finally:
            self.setattr("_booting_mixtures", False)
        return self

    def _validate_dataset_mixtures(self) -> None:
        configured_train_splits = self.get("train_splits")
        if configured_train_splits is None:
            train_splits = {"train"}
        elif isinstance(configured_train_splits, str):
            train_splits = {configured_train_splits}
        else:
            train_splits = set(configured_train_splits)
        for split, mixture in self.data.items():
            if not isinstance(mixture, DatasetMixtureConfig):
                continue
            prefix = f"data.{split}"
            if split not in train_splits:
                raise ValueError(f"dataset mixtures are only supported for training splits, not {split!r}")
            input_keys = mixture.getattr("_input_keys", frozenset(mixture))
            if "source" in input_keys:
                raise ValueError(f"{prefix} cannot define both `datasets` and `source`")
            unknown_mixture_keys = input_keys - _MIXTURE_KEYS
            if unknown_mixture_keys:
                raise ValueError(
                    f"{prefix} contains unsupported mixture keys: "
                    f"{', '.join(sorted(map(str, unknown_mixture_keys)))}"
                )
            if not isinstance(mixture.datasets, Mapping) or not mixture.datasets:
                raise ValueError(f"{prefix}.datasets must be a nonempty mapping")
            epoch_size = _resolved_value(mixture.epoch_size)
            if not mixture.getattr("_epoch_size_raw_valid") or (
                epoch_size is not None
                and (isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size <= 0)
            ):
                raise ValueError(f"{prefix}.epoch_size must be a positive integer or None")
            replacement = _resolved_value(mixture.replacement)
            if not mixture.getattr("_replacement_raw_valid") or not isinstance(replacement, bool):
                raise ValueError(f"{prefix}.replacement must be a bool")
            seed = _resolved_value(mixture.seed)
            if not mixture.getattr("_seed_raw_valid") or (
                seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0)
            ):
                raise ValueError(f"{prefix}.seed must be a nonnegative integer or None")
            _set_plain_value(mixture, "epoch_size", epoch_size)
            _set_plain_value(mixture, "replacement", replacement)
            _set_plain_value(mixture, "seed", seed)

            indices_owners: dict[Path, str] = {}
            for name, component in mixture.datasets.items():
                component_prefix = f"{prefix}.datasets.{name}"
                if not isinstance(name, str) or _DATASET_NAME_PATTERN.fullmatch(name) is None:
                    raise ValueError(
                        f"{prefix}.datasets contains invalid name {name!r}; expected [A-Za-z][A-Za-z0-9_-]*"
                    )
                component_input_keys = component.getattr("_input_keys", frozenset(component))
                if "datasets" in component_input_keys:
                    raise ValueError(f"{component_prefix} cannot define nested `datasets`")
                misplaced_controls = component_input_keys & _COMPONENT_CONTROL_KEYS
                if misplaced_controls:
                    raise ValueError(
                        f"{component_prefix} contains mixture-level controls that must be moved to {prefix}: "
                        f"{', '.join(sorted(misplaced_controls))}"
                    )
                source = _normalize_path_value(_resolved_value(component.source))
                if (
                    not component.getattr("_source_raw_valid")
                    or not isinstance(source, str)
                    or not source.strip()
                ):
                    raise ValueError(f"{component_prefix}.source must be a nonempty string or PathLike")
                _set_plain_value(component, "source", source)
                sampling_ratio = _resolved_value(component.sampling_ratio)
                if (
                    not component.getattr("_sampling_ratio_raw_valid")
                    or not _is_positive_finite_real(sampling_ratio)
                ):
                    raise ValueError(f"{component_prefix}.sampling_ratio must be finite and strictly positive")
                _set_plain_value(component, "sampling_ratio", float(sampling_ratio))
                seq_to_pdb_index = _normalize_path_value(_resolved_value(component.seq_to_pdb_index))
                msa_path = _normalize_path_value(_resolved_value(component.msa_path))
                if (seq_to_pdb_index is None) != (msa_path is None):
                    raise ValueError(
                        f"{component_prefix}.seq_to_pdb_index and {component_prefix}.msa_path must be provided together"
                    )
                if seq_to_pdb_index is not None and (
                    not component.getattr("_seq_to_pdb_index_raw_valid")
                    or not isinstance(seq_to_pdb_index, str)
                    or not seq_to_pdb_index.strip()
                ):
                    raise ValueError(
                        f"{component_prefix}.seq_to_pdb_index must be a nonempty string, PathLike, or None"
                    )
                if msa_path is not None and (
                    not component.getattr("_msa_path_raw_valid")
                    or not isinstance(msa_path, str)
                    or not msa_path.strip()
                ):
                    raise ValueError(f"{component_prefix}.msa_path must be a nonempty string, PathLike, or None")
                _set_plain_value(component, "seq_to_pdb_index", seq_to_pdb_index)
                _set_plain_value(component, "msa_path", msa_path)

                if "processing_timeout_seconds" in component_input_keys:
                    processing_timeout_seconds = _resolved_value(
                        component.get("processing_timeout_seconds")
                    )
                    if (
                        not component.getattr("_processing_timeout_seconds_raw_valid")
                        or (
                            processing_timeout_seconds is not None
                            and not _is_positive_finite_real(processing_timeout_seconds)
                        )
                    ):
                        raise ValueError(
                            f"{component_prefix}.processing_timeout_seconds must be a finite positive number or None"
                        )
                    _set_plain_value(
                        component,
                        "processing_timeout_seconds",
                        processing_timeout_seconds,
                    )
                if "max_consecutive_processing_errors" in component_input_keys:
                    max_consecutive_processing_errors = _resolved_value(
                        component.get("max_consecutive_processing_errors")
                    )
                    if (
                        not component.getattr("_max_consecutive_processing_errors_raw_valid")
                        or isinstance(max_consecutive_processing_errors, bool)
                        or (
                            max_consecutive_processing_errors is not None
                            and (
                                not isinstance(max_consecutive_processing_errors, int)
                                or max_consecutive_processing_errors <= 0
                            )
                        )
                    ):
                        raise ValueError(
                            f"{component_prefix}.max_consecutive_processing_errors "
                            "must be a positive integer or None"
                        )
                    _set_plain_value(
                        component,
                        "max_consecutive_processing_errors",
                        max_consecutive_processing_errors,
                    )

                cluster = component.cluster_sampling
                cluster_prefix = f"{component_prefix}.cluster_sampling"
                unknown_cluster_keys = cluster.getattr("_input_keys", frozenset(cluster)) - _CLUSTER_SAMPLING_KEYS
                if unknown_cluster_keys:
                    raise ValueError(
                        f"{cluster_prefix} contains unsupported keys: "
                        f"{', '.join(sorted(map(str, unknown_cluster_keys)))}"
                    )
                cluster_enabled = _resolved_value(cluster.enabled)
                if not cluster.getattr("_enabled_raw_valid") or not isinstance(cluster_enabled, bool):
                    raise ValueError(f"{cluster_prefix}.enabled must be a bool")
                _set_plain_value(cluster, "enabled", cluster_enabled)
                cluster_file = _normalize_path_value(_resolved_value(cluster.cluster_file))
                indices_file = _normalize_path_value(_resolved_value(cluster.indices_file))
                if cluster_file is not None and (
                    not cluster.getattr("_cluster_file_raw_valid")
                    or not isinstance(cluster_file, str)
                    or not cluster_file.strip()
                ):
                    raise ValueError(f"{cluster_prefix}.cluster_file must be a nonempty string, PathLike, or None")
                if indices_file is not None and (
                    not cluster.getattr("_indices_file_raw_valid")
                    or not isinstance(indices_file, str)
                    or not indices_file.strip()
                ):
                    raise ValueError(f"{cluster_prefix}.indices_file must be a nonempty string, PathLike, or None")
                if cluster_enabled and cluster_file is None:
                    raise ValueError(f"{component_prefix}.cluster_sampling.cluster_file is required when enabled")
                _set_plain_value(cluster, "cluster_file", cluster_file)
                _set_plain_value(cluster, "indices_file", indices_file)
                cluster_epoch_size = _resolved_value(cluster.epoch_size)
                cluster_seed = _resolved_value(cluster.seed)
                if cluster_epoch_size is not None:
                    raise ValueError(f"{component_prefix}.cluster_sampling.epoch_size must be None for a mixture")
                if cluster_seed is not None:
                    raise ValueError(f"{component_prefix}.cluster_sampling.seed must be None for a mixture")
                _set_plain_value(cluster, "epoch_size", None)
                _set_plain_value(cluster, "seed", None)
                cluster_replacement = _resolved_value(cluster.replacement)
                if not cluster.getattr("_replacement_raw_valid") or cluster_replacement is not True:
                    raise ValueError(f"{component_prefix}.cluster_sampling.replacement must remain True for a mixture")
                _set_plain_value(cluster, "replacement", True)
                raw_contact_cutoff = _resolved_value(cluster.contact_cutoff)
                if cluster.getattr("_contact_cutoff_is_bool") or isinstance(raw_contact_cutoff, bool):
                    raise ValueError(f"{cluster_prefix}.contact_cutoff must be finite and strictly positive")
                try:
                    contact_cutoff = float(raw_contact_cutoff)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{cluster_prefix}.contact_cutoff must be finite and strictly positive") from error
                if not math.isfinite(contact_cutoff) or contact_cutoff <= 0:
                    raise ValueError(f"{cluster_prefix}.contact_cutoff must be finite and strictly positive")
                _set_plain_value(cluster, "contact_cutoff", contact_cutoff)
                include_all_chain_pairs = _resolved_value(cluster.include_all_chain_pairs)
                if not cluster.getattr("_include_all_chain_pairs_raw_valid") or not isinstance(
                    include_all_chain_pairs, bool
                ):
                    raise ValueError(f"{cluster_prefix}.include_all_chain_pairs must be a bool")
                _set_plain_value(cluster, "include_all_chain_pairs", include_all_chain_pairs)
                if (
                    not cluster.getattr("_exclude_mol_types_is_valid")
                    or not isinstance(cluster.exclude_mol_types, (list, tuple))
                    or any(
                        not isinstance(mol_type, str) or not mol_type.strip()
                        for mol_type in cluster.exclude_mol_types
                    )
                ):
                    raise ValueError(f"{cluster_prefix}.exclude_mol_types must be a sequence of nonempty strings")
                if indices_file is not None:
                    indices_path = Path(indices_file).expanduser().resolve()
                    previous = indices_owners.setdefault(indices_path, name)
                    if previous != name:
                        raise ValueError(
                            f"{component_prefix}.cluster_sampling.indices_file duplicates the explicit cache path "
                            f"used by {prefix}.datasets.{previous}"
                        )
