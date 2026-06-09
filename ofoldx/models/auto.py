r"""
Auto factories that resolve config and model classes from an ``architecture`` key.

Mirrors the Hugging Face ``Auto*`` pattern: [`AutoConfig`][ofoldx.models.auto.AutoConfig] loads a typed
config and each ``AutoModel*`` class instantiates the model registered for that config's architecture, with
task-specific variants that additionally assert the artifact belongs to the requested task family.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from os import PathLike
from typing import Any, cast

from chanfig import Config

from ..artifacts import ARTIFACT_MANIFEST_NAME, artifact_component_path, artifact_kind, read_artifact_manifest
from ..hub import pretrained_allow_patterns, remote_code_import_path, resolve_pretrained_path
from ..processors.registry import architecture_from_config, architecture_from_config_class
from .mixin import ModelMixin, _manifest_architecture, _manifest_file, _set_dot_path
from .registry import (
    BINDING_AFFINITY_MODELS,
    DESIGN_GENERATOR_MODELS,
    MODEL_CONFIGS,
    MODELS,
    STRUCTURE_PREDICTION_MODELS,
    set_config_class_architecture,
)


class AutoConfig:
    r"""
    Resolve typed model config classes from an ``architecture`` field.

    A namespace of classmethods, not an instantiable object: the constructor raises. Use
    [`from_pretrained`][ofoldx.models.auto.AutoConfig.from_pretrained] to load a config from disk or the
    Hub and [`from_config`][ofoldx.models.auto.AutoConfig.from_config] to coerce a raw mapping into its
    registered typed class.

    Raises:
        TypeError: Always, if instantiated; use the classmethods instead.

    Examples:
        >>> from ofoldx import AutoConfig  # doctest: +SKIP
        >>> config = AutoConfig.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        raise TypeError("AutoConfig cannot be instantiated")

    @classmethod
    def register(cls, architecture: str, config_class: type[Config], *, exist_ok: bool = False) -> None:
        r"""
        Register a config class under an architecture key.

        Stamps ``architecture`` onto ``config_class`` and adds it to the config registry so that later
        ``from_config`` / ``from_pretrained`` calls can coerce raw configs to this class.

        Args:
            architecture: Public architecture key.
            config_class: Config class to instantiate for that key.
            exist_ok: If ``True``, replace an existing registration.

        Examples:
            >>> from ofoldx import AutoConfig  # doctest: +SKIP
            >>> AutoConfig.register("myarch", MyConfig)  # doctest: +SKIP
        """

        set_config_class_architecture(config_class, architecture)
        MODEL_CONFIGS.register(config_class, name=architecture, override=exist_ok)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> Config:
        r"""
        Load a typed model config from a local directory or Hub repo.

        Resolves the pretrained path (downloading from the Hub if needed), reads ``manifest.json`` and
        ``config.json``, and coerces the runtime config into its registered typed config class. If no config class is
        registered for the artifact's architecture, it falls back to the config class declared by the artifact's own
        model entrypoint.
        ``config_overrides`` are applied last, supporting dotted keys for nested config values.

        Args:
            pretrained_model_name_or_path: Local pretrained model directory or Hugging Face Hub repository id.
            config_overrides: Optional config values applied after loading ``config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Typed model config.

        Examples:
            >>> from ofoldx import AutoConfig  # doctest: +SKIP
            >>> config = AutoConfig.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> config.architecture  # doctest: +SKIP
            'alphafold3'
        """

        _ensure_default_models()
        pretrained_path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                ARTIFACT_MANIFEST_NAME,
                ModelMixin.CONFIG_NAME,
                f"*/{ARTIFACT_MANIFEST_NAME}",
                f"*/{ModelMixin.CONFIG_NAME}",
                trust_remote_code=trust_remote_code,
            ),
        )
        manifest = read_artifact_manifest(pretrained_path)
        kind = artifact_kind(manifest)
        if kind == "pipeline":
            pretrained_path = artifact_component_path(
                pretrained_path,
                manifest,
                "scorer",
                "generator",
                "predictor",
                "packer",
            )
            manifest = read_artifact_manifest(pretrained_path, expected_kind="model")
        elif kind != "model":
            raise ValueError(f"manifest.json kind {kind!r} is not 'model'")
        artifact_config = Config.from_json(pretrained_path / _manifest_file(manifest, "config"))
        artifact_config["architecture"] = _manifest_architecture(manifest)
        with remote_code_import_path(pretrained_path, trust_remote_code=trust_remote_code):
            try:
                config = MODEL_CONFIGS.coerce(artifact_config)
            except ValueError:
                model_class = ModelMixin._model_class_from_manifest(manifest)
                config = model_class._build_config(artifact_config, {}, architecture=_manifest_architecture(manifest))
        for key, value in ({} if config_overrides is None else dict(config_overrides)).items():
            _set_dot_path(config, key, value)
        return config

    @classmethod
    def from_config(cls, config: Config | Mapping[str, Any]) -> Config:
        r"""
        Return a typed model config from a raw or typed config object.

        Coerces ``config`` to the typed config class registered for its ``architecture`` field. An
        already-typed instance is re-stamped and returned as-is.

        Args:
            config: Raw config mapping or already-typed config object.

        Returns:
            Typed model config selected by the config ``architecture`` field.

        Raises:
            ValueError: If the config has no ``architecture`` field or none is registered for it.

        Examples:
            >>> from ofoldx import AutoConfig
            >>> from ofoldx.models.alphafold3 import AlphaFold3Config
            >>> config = AutoConfig.from_config(AlphaFold3Config())
            >>> config.architecture
            'alphafold3'
        """

        _ensure_default_models()
        return MODEL_CONFIGS.coerce(config)


class AutoModel:
    r"""
    Resolve and build model classes from a typed or raw config.

    A namespace of classmethods, not an instantiable object: the constructor raises. Instantiate a model in
    memory with [`from_config`][ofoldx.models.auto.AutoModel.from_config] or load a pretrained artifact
    with [`from_pretrained`][ofoldx.models.auto.AutoModel.from_pretrained].

    Raises:
        TypeError: Always, if instantiated; use the classmethods instead.

    Examples:
        >>> from ofoldx import AutoModel  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        raise TypeError("AutoModel cannot be instantiated")

    @classmethod
    def register(cls, config_class: type[Config], model_class: type[ModelMixin], *, exist_ok: bool = False) -> None:
        r"""
        Register a model class for a config class.

        Also registers ``config_class`` with [`AutoConfig`][ofoldx.models.auto.AutoConfig] so the
        architecture resolves in both directions.

        Args:
            config_class: Config class declaring ``architecture``.
            model_class: Model class to instantiate.
            exist_ok: If ``True``, replace existing registrations.

        Raises:
            TypeError: If ``model_class`` does not inherit [`ModelMixin`][ofoldx.models.mixin.ModelMixin].

        Examples:
            >>> from ofoldx import AutoModel  # doctest: +SKIP
            >>> AutoModel.register(MyConfig, MyModel)  # doctest: +SKIP
        """

        _register_model(config_class, model_class, MODELS, exist_ok=exist_ok)

    @classmethod
    def from_config(cls, config: Config | Mapping[str, Any]) -> ModelMixin:
        r"""
        Build a model from a raw or typed config object.

        Coerces ``config`` to its typed class and instantiates the model registered for its architecture. The
        model is returned with freshly initialized (random) weights.

        Args:
            config: Raw config mapping or already-typed config object.

        Returns:
            Instantiated model registered for the config ``architecture``.

        Raises:
            ValueError: If the config has no ``architecture`` field or none is registered for it.

        Examples:
            >>> from ofoldx import AutoModel  # doctest: +SKIP
            >>> from ofoldx.models.alphafold3 import AlphaFold3Config  # doctest: +SKIP
            >>> model = AutoModel.from_config(AlphaFold3Config())  # doctest: +SKIP
        """

        _ensure_default_models()
        return MODELS.build(AutoConfig.from_config(config))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> ModelMixin:
        r"""
        Load any OFoldX model from a local directory or Hub repo.

        Resolves the artifact, reads its ``manifest.json`` and ``config.json``, selects the model class recorded in the
        manifest ``entrypoint`` field, builds it, and loads ``model.safetensors``. Delegates to
        [`ModelMixin.from_pretrained`][ofoldx.models.mixin.ModelMixin.from_pretrained].

        Args:
            pretrained_model_name_or_path: Local pretrained model directory or Hugging Face Hub repository id.
            strict: Whether checkpoint tensors must exactly match model parameters.
            weights_device: Device used while materializing safetensors weights.
            variant: Optional model weight variant suffix.
            config_overrides: Optional config values applied after loading ``config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Loaded model instance.

        Examples:
            >>> from ofoldx import AutoModel, AutoProcessor  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> output = model(processor("complex.cif"))  # doctest: +SKIP
        """

        _ensure_default_models()
        return cast(
            ModelMixin,
            ModelMixin.from_pretrained(
                pretrained_model_name_or_path,
                strict=strict,
                weights_device=weights_device,
                variant=variant,
                config_overrides=config_overrides,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                trust_remote_code=trust_remote_code,
            ),
        )


class AutoModelForStructurePrediction:
    r"""
    Resolve and build structure-prediction model classes from a typed or raw config.

    A task-specific variant of [`AutoModel`][ofoldx.models.auto.AutoModel] that only resolves
    architectures registered for structure prediction and, on
    [`from_pretrained`][ofoldx.models.auto.AutoModelForStructurePrediction.from_pretrained], verifies the
    loaded artifact belongs to that task family. A namespace of classmethods; the constructor raises.

    Raises:
        TypeError: Always, if instantiated; use the classmethods instead.

    Examples:
        >>> from ofoldx import AutoModelForStructurePrediction  # doctest: +SKIP
        >>> model = AutoModelForStructurePrediction.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        raise TypeError("AutoModelForStructurePrediction cannot be instantiated")

    @classmethod
    def register(cls, config_class: type[Config], model_class: type[ModelMixin], *, exist_ok: bool = False) -> None:
        r"""
        Register a structure-prediction model class for a config class.

        Registers the pair in both the generic model registry and the structure-prediction registry, so the
        architecture resolves through [`AutoModel`][ofoldx.models.auto.AutoModel] and this task-specific
        factory alike.

        Args:
            config_class: Config class declaring ``architecture``.
            model_class: Structure-prediction model class to instantiate.
            exist_ok: If ``True``, replace existing registrations.

        Raises:
            TypeError: If ``model_class`` does not inherit [`ModelMixin`][ofoldx.models.mixin.ModelMixin].

        Examples:
            >>> from ofoldx import AutoModelForStructurePrediction  # doctest: +SKIP
            >>> AutoModelForStructurePrediction.register(MyConfig, MyModel)  # doctest: +SKIP
        """

        _register_model(config_class, model_class, MODELS, exist_ok=exist_ok)
        _register_model(config_class, model_class, STRUCTURE_PREDICTION_MODELS, exist_ok=exist_ok)

    @classmethod
    def from_config(cls, config: Config | Mapping[str, Any]) -> ModelMixin:
        r"""
        Build a structure-prediction model from a raw or typed config object.

        Args:
            config: Raw config mapping or already-typed config object.

        Returns:
            Structure-prediction model registered for the config ``architecture``.

        Raises:
            ValueError: If the config has no ``architecture`` field or none is registered for structure
                prediction.

        Examples:
            >>> from ofoldx import AutoModelForStructurePrediction  # doctest: +SKIP
            >>> from ofoldx.models.alphafold3 import AlphaFold3Config  # doctest: +SKIP
            >>> model = AutoModelForStructurePrediction.from_config(AlphaFold3Config())  # doctest: +SKIP
        """

        _ensure_default_models()
        return STRUCTURE_PREDICTION_MODELS.build(AutoConfig.from_config(config))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> ModelMixin:
        r"""
        Load a registered structure-prediction model from a local directory or Hub repo.

        Loads the artifact via [`AutoModel.from_pretrained`][ofoldx.models.auto.AutoModel.from_pretrained],
        then verifies that its architecture and concrete class are registered for structure prediction,
        raising otherwise. See [`AutoModel.from_pretrained`][ofoldx.models.auto.AutoModel.from_pretrained]
        for the full argument reference.

        Args:
            pretrained_model_name_or_path: Local pretrained model directory or Hugging Face Hub repository id.
            strict: Whether checkpoint tensors must exactly match model parameters.
            weights_device: Device used while materializing safetensors weights.
            variant: Optional model weight variant suffix.
            config_overrides: Optional config values applied after loading ``config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Loaded structure-prediction model instance.

        Raises:
            TypeError: If the loaded artifact's architecture or class is not registered for structure
                prediction.

        Examples:
            >>> from ofoldx import AutoModelForStructurePrediction  # doctest: +SKIP
            >>> model = AutoModelForStructurePrediction.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        """

        _ensure_default_models()
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            strict=strict,
            weights_device=weights_device,
            variant=variant,
            config_overrides=config_overrides,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        architecture = architecture_from_config(model.config)
        if architecture not in STRUCTURE_PREDICTION_MODELS:
            raise TypeError(
                f"pretrained artifact architecture {architecture!r} is not registered for structure prediction"
            )
        expected_cls = cast(type[ModelMixin], STRUCTURE_PREDICTION_MODELS.lookup(architecture))
        if type(model) is not expected_cls:
            raise TypeError(
                f"pretrained artifact class {type(model).__name__} is not registered for structure prediction "
                f"under architecture {architecture!r}"
            )
        return model


class AutoModelForDesignGeneration:
    r"""
    Resolve and build design-generator model classes from a typed or raw config.

    A task-specific variant of [`AutoModel`][ofoldx.models.auto.AutoModel] that only resolves
    architectures registered for design generation and, on
    [`from_pretrained`][ofoldx.models.auto.AutoModelForDesignGeneration.from_pretrained], verifies the
    loaded artifact belongs to that task family. A namespace of classmethods; the constructor raises.

    Raises:
        TypeError: Always, if instantiated; use the classmethods instead.

    Examples:
        >>> from ofoldx import AutoModelForDesignGeneration  # doctest: +SKIP
        >>> generator = AutoModelForDesignGeneration.from_pretrained("oteam/proteinmpnn")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        raise TypeError("AutoModelForDesignGeneration cannot be instantiated")

    @classmethod
    def register(cls, config_class: type[Config], model_class: type[ModelMixin], *, exist_ok: bool = False) -> None:
        r"""
        Register a design-generator model class for a config class.

        Registers the pair in both the generic model registry and the design-generator registry, so the
        architecture resolves through [`AutoModel`][ofoldx.models.auto.AutoModel] and this task-specific
        factory alike.

        Args:
            config_class: Config class declaring ``architecture``.
            model_class: Design-generator model class to instantiate.
            exist_ok: If ``True``, replace existing registrations.

        Raises:
            TypeError: If ``model_class`` does not inherit [`ModelMixin`][ofoldx.models.mixin.ModelMixin].

        Examples:
            >>> from ofoldx import AutoModelForDesignGeneration  # doctest: +SKIP
            >>> AutoModelForDesignGeneration.register(MyConfig, MyGenerator)  # doctest: +SKIP
        """

        _register_model(config_class, model_class, MODELS, exist_ok=exist_ok)
        _register_model(config_class, model_class, DESIGN_GENERATOR_MODELS, exist_ok=exist_ok)

    @classmethod
    def from_config(cls, config: Config | Mapping[str, Any]) -> ModelMixin:
        r"""
        Build a design-generator model from a raw or typed config object.

        Args:
            config: Raw config mapping or already-typed config object.

        Returns:
            Design-generator model registered for the config ``architecture``.

        Raises:
            ValueError: If the config has no ``architecture`` field or none is registered for design
                generation.

        Examples:
            >>> from ofoldx import AutoModelForDesignGeneration  # doctest: +SKIP
            >>> from ofoldx.models.proteinmpnn import ProteinMPNNConfig  # doctest: +SKIP
            >>> generator = AutoModelForDesignGeneration.from_config(ProteinMPNNConfig())  # doctest: +SKIP
        """

        _ensure_default_models()
        return DESIGN_GENERATOR_MODELS.build(AutoConfig.from_config(config))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> ModelMixin:
        r"""
        Load a registered design-generator model from a local directory or Hub repo.

        Loads the artifact via [`AutoModel.from_pretrained`][ofoldx.models.auto.AutoModel.from_pretrained],
        then verifies that its architecture and concrete class are registered for design generation, raising
        otherwise. See [`AutoModel.from_pretrained`][ofoldx.models.auto.AutoModel.from_pretrained] for the
        full argument reference.

        Args:
            pretrained_model_name_or_path: Local pretrained model directory or Hugging Face Hub repository id.
            strict: Whether checkpoint tensors must exactly match model parameters.
            weights_device: Device used while materializing safetensors weights.
            variant: Optional model weight variant suffix.
            config_overrides: Optional config values applied after loading ``config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Loaded design-generator model instance.

        Raises:
            TypeError: If the loaded artifact's architecture or class is not registered for design generation.

        Examples:
            >>> from ofoldx import AutoModelForDesignGeneration  # doctest: +SKIP
            >>> generator = AutoModelForDesignGeneration.from_pretrained("oteam/proteinmpnn")  # doctest: +SKIP
        """

        _ensure_default_models()
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            strict=strict,
            weights_device=weights_device,
            variant=variant,
            config_overrides=config_overrides,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        architecture = architecture_from_config(model.config)
        if architecture not in DESIGN_GENERATOR_MODELS:
            raise TypeError(
                f"pretrained artifact architecture {architecture!r} is not registered for design generation"
            )
        expected_cls = cast(type[ModelMixin], DESIGN_GENERATOR_MODELS.lookup(architecture))
        if type(model) is not expected_cls:
            raise TypeError(
                f"pretrained artifact class {type(model).__name__} is not registered for design generation "
                f"under architecture {architecture!r}"
            )
        return model


class AutoModelForBindingAffinityPrediction:
    r"""
    Resolve and build binding-affinity prediction model classes from a typed or raw config.
    """

    def __init__(self) -> None:
        raise TypeError("AutoModelForBindingAffinityPrediction cannot be instantiated")

    @classmethod
    def register(cls, config_class: type[Config], model_class: type[ModelMixin], *, exist_ok: bool = False) -> None:
        r"""
        Register a binding-affinity prediction model class for a config class.
        """

        _register_model(config_class, model_class, MODELS, exist_ok=exist_ok)
        _register_model(config_class, model_class, BINDING_AFFINITY_MODELS, exist_ok=exist_ok)

    @classmethod
    def from_config(cls, config: Config | Mapping[str, Any]) -> ModelMixin:
        r"""
        Build a binding-affinity prediction model from a raw or typed config object.
        """

        _ensure_default_models()
        return BINDING_AFFINITY_MODELS.build(AutoConfig.from_config(config))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> ModelMixin:
        r"""
        Load a registered binding-affinity prediction model from a local directory or Hub repo.
        """

        _ensure_default_models()
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            strict=strict,
            weights_device=weights_device,
            variant=variant,
            config_overrides=config_overrides,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        architecture = architecture_from_config(model.config)
        if architecture not in BINDING_AFFINITY_MODELS:
            raise TypeError(
                f"pretrained artifact architecture {architecture!r} is not registered for binding-affinity prediction"
            )
        expected_cls = cast(type[ModelMixin], BINDING_AFFINITY_MODELS.lookup(architecture))
        if type(model) is not expected_cls:
            raise TypeError(
                f"pretrained artifact class {type(model).__name__} is not registered for binding-affinity prediction "
                f"under architecture {architecture!r}"
            )
        return model


def _ensure_default_models() -> None:
    r"""
    Import every built-in architecture module so its classes self-register.

    Idempotent: each import triggers the module's registration side effects exactly once. Called at the start
    of every ``Auto*`` resolution so the default architectures are always available.
    """

    for module_name in (
        ".alphafold2",
        ".alphafold3",
        ".boltz1",
        ".boltz2",
        ".boltzgen",
        ".esmfold2",
        ".ligandmpnn",
        ".odesign",
        ".proteinmpnn",
        ".protenix",
        ".pxdesign",
        ".rfdiffusion",
    ):
        importlib.import_module(module_name, package=__package__)


def _register_model(
    config_class: type[Config],
    model_class: type[ModelMixin],
    registry: Any,
    *,
    exist_ok: bool,
) -> None:
    r"""
    Register a config/model pair into a registry under the config's architecture.

    Also registers ``config_class`` with [`AutoConfig`][ofoldx.models.auto.AutoConfig] so the
    architecture resolves both to a config and to a model.

    Args:
        config_class: Config class declaring the architecture.
        model_class: Model class to register; must inherit [`ModelMixin`][ofoldx.models.mixin.ModelMixin].
        registry: Target model registry.
        exist_ok: If ``True``, replace an existing registration in ``registry``.

    Raises:
        TypeError: If ``model_class`` does not inherit [`ModelMixin`][ofoldx.models.mixin.ModelMixin].
    """

    if not issubclass(model_class, ModelMixin):
        raise TypeError(f"{model_class.__name__} must inherit ModelMixin")
    architecture = architecture_from_config_class(config_class)
    AutoConfig.register(architecture, config_class, exist_ok=True)
    registry.register(model_class, name=architecture, override=exist_ok)
