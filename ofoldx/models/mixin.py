r"""
Model-side save/load protocol shared by every OFoldX architecture.

[`ModelMixin`][ofoldx.models.mixin.ModelMixin] gives models a config-bound constructor, deferred
parameter initialization, and ``save_pretrained`` / ``from_pretrained`` / ``push_to_hub`` round-trips over
``manifest.json`` + ``config.json`` + ``model.safetensors`` artifacts. The manifest records a fully qualified
``entrypoint`` so ``from_pretrained`` can recover the exact model class without a central registry.
"""

from __future__ import annotations

import importlib
import warnings
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Iterator, Mapping, Protocol, Sequence, TypeVar, cast

from chanfig import Config
from oops import LayerNorm
from safetensors.torch import load_file, save_file
from torch import nn

from ..artifacts import (
    ARTIFACT_MANIFEST_NAME,
    artifact_component_path,
    artifact_kind,
    build_artifact_manifest,
    entrypoint,
    read_artifact_manifest,
)
from ..hub import pretrained_allow_patterns, publish_artifact_to_hub, remote_code_import_path, resolve_pretrained_path
from ..processors.registry import architecture_from_config, architecture_from_config_class
from .registry import set_architecture

T = TypeVar("T", bound="ModelMixin")
_INITIALIZE_WEIGHTS = ContextVar("ofoldx_initialize_weights", default=True)


class LoadStateDictResult(Protocol):
    r"""
    Structural type of the ``NamedTuple`` returned by ``nn.Module.load_state_dict``.

    Captures only the two fields this module inspects when reporting a non-strict load, so type checkers
    accept the torch return value without depending on its exact class.

    Attributes:
        missing_keys: Parameter names present in the model but absent from the checkpoint.
        unexpected_keys: Checkpoint keys with no matching model parameter.
    """

    missing_keys: Sequence[str]
    unexpected_keys: Sequence[str]


@contextmanager
def no_init_weights() -> Iterator[None]:
    r"""
    Skip model parameter initialization inside the context.

    ``from_pretrained`` uses this for strict loads: since every parameter is about to be overwritten by the
    checkpoint, running the (potentially expensive) initialization first would be wasted work. Constructors
    consult this flag in [`post_init`][ofoldx.models.mixin.ModelMixin.post_init]. The flag is restored on
    exit even if the body raises.

    Yields:
        ``None`` while model constructors should avoid initializing parameters that will be strictly loaded from a
        checkpoint.

    Examples:
        >>> from ofoldx.models.mixin import no_init_weights
        >>> with no_init_weights():
        ...     pass  # construct a model here whose weights will be strictly loaded
    """

    token = _INITIALIZE_WEIGHTS.set(False)
    try:
        yield
    finally:
        _INITIALIZE_WEIGHTS.reset(token)


class ModelMixin(nn.Module):
    r"""
    Base ``nn.Module`` giving every OFoldX model a config-bound constructor and pretrained IO.

    Subclasses set the ``config_class`` class attribute, build their submodules in ``__init__``, and call
    [`post_init`][ofoldx.models.mixin.ModelMixin.post_init] once construction is complete to initialize
    parameters. The mixin then supplies ``save_pretrained`` / ``from_pretrained`` / ``push_to_hub`` over the
    standard ``manifest.json`` + ``config.json`` + ``model.safetensors`` artifact layout.

    Args:
        config: Runtime model config; bound to ``self.config``.

    Attributes:
        CONFIG_NAME: Filename of the serialized config inside an artifact directory.
        WEIGHTS_NAME: Default filename of the serialized weights inside an artifact directory.
        config_class: Config class a subclass instantiates from a raw artifact config.

    Examples:
        >>> from ofoldx import AutoModel  # doctest: +SKIP
        >>> model = AutoModel.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        >>> model.save_pretrained("./local-copy")  # doctest: +SKIP
    """

    CONFIG_NAME: ClassVar[str] = "config.json"
    WEIGHTS_NAME: ClassVar[str] = "model.safetensors"
    config_class: ClassVar[type[Config]]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def post_init(self) -> None:
        r"""
        Initialize parameters after all submodules have been constructed.

        Subclasses call this at the end of ``__init__``. It applies the default per-module initialization to
        every submodule and then the model-level
        [`reset_parameters`][ofoldx.models.mixin.ModelMixin.reset_parameters] hook. Inside a
        [`no_init_weights`][ofoldx.models.mixin.no_init_weights] context it returns immediately, leaving
        parameters uninitialized for an imminent strict checkpoint load.
        """

        if not _INITIALIZE_WEIGHTS.get():
            return
        self.apply(self._init_module_weights)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Apply model-level parameter initialization after the shared per-module sweep.

        The base implementation is a no-op. Subclasses override it to set values the generic sweep cannot
        infer, such as zero-initialized recycle gates or other structured initializations.
        """

    def _init_module_weights(self, module: nn.Module) -> None:
        r"""
        Apply the default OFoldX initialization to a single submodule.

        Truncated-normal for trainable linear weights (zeros for their bias) and ones/zeros for layer-norm
        weight/bias. Frozen linear weights are left untouched so deliberate fixed projections survive.

        Args:
            module: Submodule to initialize; only ``nn.Linear`` and layer-norm modules are affected.

        Raises:
            TypeError: If the config's ``initializer_range`` is not numeric.
        """

        if isinstance(module, nn.Linear):
            # A frozen weight carries a deliberate value (e.g. FrozenFourierEmbedding's normal(0,1) projection);
            # the generic sweep must never overwrite it.
            if not module.weight.requires_grad:
                return
            initializer_range = self.config.get("initializer_range", 0.02)
            if not isinstance(initializer_range, int | float):
                raise TypeError("initializer_range must be numeric")
            nn.init.trunc_normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def save_pretrained(self, save_directory: str | PathLike[str], *, variant: str | None = None) -> None:
        r"""
        Save ``manifest.json``, ``config.json``, and ``model.safetensors`` into a directory.

        Writes a self-describing artifact: the manifest records the architecture, a fully qualified
        ``entrypoint`` class path, and the weights/config files, so
        [`from_pretrained`][ofoldx.models.mixin.ModelMixin.from_pretrained] can reconstruct the exact
        model class. The directory is created if it does not exist.

        Args:
            save_directory: Destination directory.
            variant: Optional weight filename variant. ``None`` writes ``model.safetensors``.

        Examples:
            >>> from ofoldx import AutoModel  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> model.save_pretrained("./local-copy")  # doctest: +SKIP
            >>> from tempfile import TemporaryDirectory
            >>> class DemoConfig(Config):
            ...     architecture = "demo_model"
            >>> class DemoModel(ModelMixin):
            ...     config_class = DemoConfig
            ...     def __init__(self, config):
            ...         super().__init__(config)
            ...         self.linear = nn.Linear(1, 1)
            >>> with TemporaryDirectory() as directory:
            ...     DemoModel(DemoConfig({"hidden_size": 1})).save_pretrained(directory)
            ...     manifest = Config.from_json(Path(directory) / "manifest.json")
            ...     runtime_config = Config.from_json(Path(directory) / "config.json")
            ...     (manifest["kind"], manifest["architecture"], sorted(runtime_config.items()))
            ('model', 'demo_model', [('hidden_size', 1)])
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._manifest_for_save(variant).json(save_directory / ARTIFACT_MANIFEST_NAME)
        self._config_for_save().json(save_directory / self.CONFIG_NAME)
        save_file(dict(self.state_dict()), str(save_directory / self._weights_name(variant)))

    def push_to_hub(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        private: bool | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        create_pr: bool = False,
        delete_existing: bool = False,
        tag: str | None = None,
        model_card: bool = True,
        model_card_metadata: Mapping[str, object] | None = None,
        widget_examples: Sequence[Mapping[str, object]] | None = None,
        card_body: str | None = None,
        variant: str | None = None,
    ) -> Any:
        r"""
        Save this model artifact and upload it to the Hugging Face Hub.

        Serializes the model into a temporary directory via
        [`save_pretrained`][ofoldx.models.mixin.ModelMixin.save_pretrained] and uploads it as a single
        commit, so the Hub repo mirrors the local artifact layout.

        Args:
            repo_id: Destination Hugging Face Hub repository id.
            repo_type: Hub repository type.
            private: Optional privacy setting used when creating the repository.
            token: Optional Hugging Face authentication token.
            revision: Optional target branch for the upload.
            commit_message: Optional commit message for the upload.
            create_pr: If ``True``, upload changes through a Hub pull request.
            delete_existing: If ``True``, delete an existing destination repo before recreating it.
            tag: Optional Hub tag to create after uploading.
            model_card: If ``True``, write/update ``README.md`` with Hub card metadata.
            model_card_metadata: Extra model-card front matter values.
            widget_examples: Optional widget examples. ``None`` uses task defaults.
            card_body: Optional Markdown body for the generated model card.
            variant: Optional weight filename variant to save and upload.

        Returns:
            The object returned by ``huggingface_hub.HfApi.upload_folder``.

        Examples:
            >>> from ofoldx import AutoModel  # doctest: +SKIP
            >>> model = AutoModel.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> model.push_to_hub("oteam/alphafold3")  # doctest: +SKIP
        """

        with TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, variant=variant)
            return publish_artifact_to_hub(
                tmpdir,
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                token=token,
                revision=revision,
                commit_message=commit_message,
                create_pr=create_pr,
                delete_existing=delete_existing,
                tag=tag,
                model_card=model_card,
                model_card_metadata=model_card_metadata,
                widget_examples=widget_examples,
                card_body=card_body,
            )

    @classmethod
    def from_pretrained(
        cls: type[T],
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
    ) -> T:
        r"""
        Load a model from a local OFoldX pretrained directory.

        ``weights_device`` controls where safetensors materializes checkpoint tensors before
        ``load_state_dict`` copies them into the newly constructed model.

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

        Raises:
            NotADirectoryError: If the resolved pretrained path is not a directory.
            ValueError: If the saved config schema or class metadata is invalid.

        Examples:
            >>> from ofoldx.models.alphafold3 import AlphaFold3Model  # doctest: +SKIP
            >>> model = AlphaFold3Model.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        """

        pretrained_path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                ARTIFACT_MANIFEST_NAME,
                cls.CONFIG_NAME,
                cls._weights_name(variant),
                f"*/{ARTIFACT_MANIFEST_NAME}",
                f"*/{cls.CONFIG_NAME}",
                f"*/{cls._weights_name(variant)}",
                trust_remote_code=trust_remote_code,
            ),
        )
        if not pretrained_path.is_dir():
            raise NotADirectoryError(f"pretrained model path must be a directory: {pretrained_path}")

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
        with remote_code_import_path(pretrained_path, trust_remote_code=trust_remote_code):
            model_cls = cls._model_class_from_manifest(manifest)
            config_path = pretrained_path / _manifest_file(manifest, "config")
            artifact_config = Config.from_json(config_path)
            model_config = model_cls._build_config(
                artifact_config,
                {} if config_overrides is None else dict(config_overrides),
                architecture=_manifest_architecture(manifest),
            )
        # A strict load overwrites every parameter, so skip the about-to-be-discarded initialization. A
        # non-strict load may leave parameters absent from the checkpoint, so initialize normally first to
        # guarantee those keep their real init (e.g. zero-initialized recycle gates) rather than torch defaults.
        with no_init_weights() if strict else nullcontext():
            model = model_cls(model_config)
        state_dict = load_file(str(pretrained_path / _manifest_weight(manifest, variant)), device=weights_device)
        load_result = model.load_state_dict(dict(state_dict), strict=strict)
        model._report_load_result(load_result, source=str(pretrained_path))
        return cast(T, model)

    @classmethod
    def _weights_name(cls, variant: str | None) -> str:
        r"""
        Return the weights filename for a variant.

        Args:
            variant: Optional variant suffix; ``None`` selects the default ``WEIGHTS_NAME``.

        Returns:
            The safetensors filename, e.g. ``"model.safetensors"`` or ``"model.<variant>.safetensors"``.
        """

        if variant is None:
            return cls.WEIGHTS_NAME
        return f"model.{variant}.safetensors"

    def _report_load_result(self, load_result: LoadStateDictResult, *, source: str) -> None:
        r"""
        Warn about freshly-initialized and ignored checkpoint tensors.

        Args:
            load_result: Result of ``load_state_dict``, providing missing and unexpected key lists.
            source: Human-readable artifact source, used in the warning messages.
        """

        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        if missing:
            warnings.warn(
                f"{type(self).__name__}.from_pretrained({source}): {len(missing)} tensor(s) "
                f"absent from the checkpoint, left at initialized values, e.g. {missing[:5]}",
                stacklevel=3,
            )
        if unexpected:
            warnings.warn(
                f"{type(self).__name__}.from_pretrained({source}): {len(unexpected)} checkpoint "
                f"tensor(s) had no matching parameter and were ignored, e.g. {unexpected[:5]}",
                stacklevel=3,
            )

    def _config_for_save(self) -> Config:
        r"""
        Build the runtime config to serialize.

        Returns:
            A copy of ``self.config`` without artifact-only metadata.
        """

        config = self.config.copy()
        for key in ("architecture", "entrypoint", "producer", "format", "format_version", "kind", "task"):
            config.pop(key, None)
        return config

    def _manifest_for_save(self, variant: str | None) -> Config:
        r"""
        Build the root artifact manifest for this model.

        Args:
            variant: Optional weight filename variant.

        Returns:
            Manifest config written to ``manifest.json``.
        """

        try:
            architecture = architecture_from_config(self.config)
        except ValueError:
            architecture = architecture_from_config_class(type(self).config_class)
        task = _task_for_model(type(self), architecture)
        return build_artifact_manifest(
            kind="model",
            entrypoint=_qualified_class_name(type(self)),
            architecture=architecture,
            task=task,
            files={"config": self.CONFIG_NAME},
            weights={_weights_manifest_key(variant): self._weights_name(variant)},
        )

    @classmethod
    def _model_class_from_manifest(cls: type[T], manifest: Config) -> type[ModelMixin]:
        r"""
        Resolve the model class recorded in an artifact manifest's ``entrypoint`` field.

        Args:
            manifest: Artifact manifest expected to carry a fully qualified ``entrypoint`` class path.

        Returns:
            The resolved [`ModelMixin`][ofoldx.models.mixin.ModelMixin] subclass.

        Raises:
            ValueError: If ``entrypoint`` is missing or not fully qualified.
            TypeError: If a concrete subclass is loading an artifact for a different model class.
        """

        model_cls = _resolve_model_class(entrypoint(manifest))
        if cls is not ModelMixin and model_cls is not cls:
            raise TypeError(f"{cls.__name__}.from_pretrained cannot load artifact for {model_cls.__name__}")
        return model_cls

    @classmethod
    def _build_config(
        cls,
        artifact_config: Config,
        config_overrides: dict[str, object],
        *,
        architecture: str | None = None,
    ) -> Config:
        r"""
        Coerce a raw artifact config into the model's typed config, applying overrides.

        Args:
            artifact_config: Raw config loaded from the artifact's ``config.json``.
            config_overrides: Dotted-key overrides applied after coercion; each emits a warning.
            architecture: Optional architecture key from the artifact manifest.

        Returns:
            The typed config with artifact-only metadata stripped and overrides applied.
        """

        config = cls.config_class(artifact_config.copy())
        if architecture is not None:
            set_architecture(config, architecture)
        for key, value in config_overrides.items():
            warnings.warn(f"overriding config value {key!r} while loading pretrained model", stacklevel=2)
            _set_dot_path(config, key, value)
        return config


def _qualified_class_name(cls: type[object]) -> str:
    r"""
    Return the fully qualified ``module.qualname`` for a class.

    Args:
        cls: Class to describe.

    Returns:
        The dotted import path suitable for later resolution.
    """

    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_model_class(class_name: str) -> type[ModelMixin]:
    r"""
    Import and return a [`ModelMixin`][ofoldx.models.mixin.ModelMixin] subclass by its dotted path.

    Args:
        class_name: Fully qualified ``module.qualname`` of the target class.

    Returns:
        The resolved model class.

    Raises:
        ValueError: If ``class_name`` is not fully qualified.
        TypeError: If the path does not resolve to a [`ModelMixin`][ofoldx.models.mixin.ModelMixin]
            subclass.
    """

    module_name, _, attr_name = class_name.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"entrypoint must be fully qualified: {class_name!r}")
    module = importlib.import_module(module_name)
    target: object = module
    for part in attr_name.split("."):
        target = getattr(target, part)
    if not isinstance(target, type):
        raise TypeError(f"entrypoint does not resolve to a class: {class_name!r}")
    if not issubclass(target, ModelMixin):
        raise TypeError(f"{class_name} is not an OFoldX ModelMixin subclass")
    return target


def _manifest_architecture(manifest: Config) -> str:
    r"""
    Return the architecture declared by a model manifest.

    Args:
        manifest: Parsed model artifact manifest.

    Returns:
        Lowercase architecture key.

    Raises:
        ValueError: If the manifest does not declare an architecture.
    """

    architecture = manifest.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("manifest.json must contain a model architecture")
    return architecture.lower()


def _manifest_file(manifest: Config, name: str) -> str:
    r"""
    Return a named file path from a model manifest.

    Args:
        manifest: Parsed model artifact manifest.
        name: File role to look up.

    Returns:
        Relative file path.
    """

    files = manifest.get("files")
    if not isinstance(files, Config):
        raise ValueError("manifest.json must contain a files mapping")
    path = files.get(name)
    if not isinstance(path, str) or not path:
        raise ValueError(f"manifest.json files must contain {name!r}")
    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"manifest.json file {name!r} path must stay inside the artifact")
    return path


def _manifest_weight(manifest: Config, variant: str | None) -> str:
    r"""
    Return the weights file path for ``variant`` from a model manifest.

    Args:
        manifest: Parsed model artifact manifest.
        variant: Optional weights variant.

    Returns:
        Relative weights file path.
    """

    weights = manifest.get("weights")
    if not isinstance(weights, Config):
        raise ValueError("manifest.json must contain a weights mapping")
    key = _weights_manifest_key(variant)
    path = weights.get(key)
    if not isinstance(path, str) or not path:
        raise ValueError(f"manifest.json weights must contain {key!r}")
    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"manifest.json weights {key!r} path must stay inside the artifact")
    return path


def _weights_manifest_key(variant: str | None) -> str:
    r"""
    Return the manifest key used for a weights variant.
    """

    return "default" if variant is None else variant


def _task_for_model(model_cls: type[ModelMixin], architecture: str) -> str | None:
    r"""
    Return the registered task for ``model_cls`` when it is known.

    Args:
        model_cls: Model class being saved.
        architecture: Canonical architecture key.

    Returns:
        A canonical task name, or ``None`` when the model has no task-level facade yet.
    """

    from .registry import BINDING_AFFINITY_MODELS, DESIGN_GENERATOR_MODELS, STRUCTURE_PREDICTION_MODELS

    if architecture == "ligandmpnn_packer":
        return "side_chain_packing"

    registries = (
        ("structure_prediction", STRUCTURE_PREDICTION_MODELS),
        ("design_generation", DESIGN_GENERATOR_MODELS),
        ("binding_affinity_prediction", BINDING_AFFINITY_MODELS),
    )
    for task, registry in registries:
        try:
            registered_cls = registry.lookup(architecture)
        except (KeyError, ValueError):
            continue
        if registered_cls is model_cls:
            return task
    return None


def _set_dot_path(config: Config, key: str, value: object) -> None:
    r"""
    Set a possibly nested config value addressed by a dotted key, in place.

    Args:
        config: Config to mutate.
        key: Dotted path (e.g. ``"trunk.num_pairformer_blocks"``); intermediate configs are created as needed.
        value: Value to assign at the leaf.

    Raises:
        TypeError: If an intermediate segment resolves to a non-config value.
    """

    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, Config())
        if not isinstance(child, Config):
            raise TypeError(f"cannot override nested config value through non-config key {part!r}")
        target = child
    target[parts[-1]] = value
