r"""
Processor save/load protocol.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, TypeVar, cast

from chanfig import Config
from typing_extensions import Self

from ..artifacts import (
    ARTIFACT_MANIFEST_NAME,
    artifact_component_path,
    artifact_kind,
    build_artifact_manifest,
    entrypoint,
    read_artifact_manifest,
)
from ..hub import pretrained_allow_patterns, publish_artifact_to_hub, remote_code_import_path, resolve_pretrained_path

T = TypeVar("T", bound="ProcessorMixin")


class ProcessorMixin:
    r"""
    Save/load mixin giving OFoldX processors a Hugging Face-style pretrained round-trip.

    A processor persists as a root ``manifest.json`` plus a ``processor_config.json`` holding its constructor
    kwargs. ``from_pretrained`` resolves the manifest entrypoint and reinstantiates that class. Subclasses implement
    ``_processor_config`` to declare the kwargs to persist.

    Attributes:
        CONFIG_NAME: File name of the persisted processor config.

    Examples:
        >>> processor.save_pretrained("./my_processor")  # doctest: +SKIP
        >>> loaded = ProcessorMixin.from_pretrained("./my_processor")  # doctest: +SKIP
    """

    CONFIG_NAME: ClassVar[str] = "processor_config.json"

    def save_pretrained(self, save_directory: str | PathLike[str]) -> None:
        r"""
        Save ``processor_config.json`` into a directory.

        Args:
            save_directory: Destination directory.

        Examples:
            >>> processor.save_pretrained("./my_processor")  # doctest: +SKIP
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._manifest_for_save().json(save_directory / ARTIFACT_MANIFEST_NAME)
        self._config_for_save().json(save_directory / self.CONFIG_NAME)

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
    ) -> Any:
        r"""
        Save this processor artifact and upload it to the Hugging Face Hub.

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

        Returns:
            The object returned by ``huggingface_hub.HfApi.upload_folder``.

        Examples:
            >>> processor.push_to_hub("oteam/my-processor")  # doctest: +SKIP
        """

        with TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
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
        pretrained_processor_name_or_path: str | PathLike[str],
        *,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> T:
        r"""
        Load a processor from a local OFoldX processor directory.

        Args:
            pretrained_processor_name_or_path: Local processor directory or Hugging Face Hub repository id.
            config_overrides: Optional config values applied after loading ``processor_config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Loaded processor instance.

        Raises:
            NotADirectoryError: If the resolved pretrained path is not a directory.
            TypeError: If called on a concrete processor class with an artifact for a different class.
            ValueError: If the saved config schema or class metadata is invalid.

        Examples:
            >>> from ofoldx.processors import ProcessorMixin  # doctest: +SKIP
            >>> processor = ProcessorMixin.from_pretrained("oteam/my-processor")  # doctest: +SKIP
        """

        pretrained_path = resolve_pretrained_path(
            pretrained_processor_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                ARTIFACT_MANIFEST_NAME,
                cls.CONFIG_NAME,
                f"*/{ARTIFACT_MANIFEST_NAME}",
                f"*/{cls.CONFIG_NAME}",
                trust_remote_code=trust_remote_code,
            ),
        )
        if not pretrained_path.is_dir():
            raise NotADirectoryError(f"pretrained processor path must be a directory: {pretrained_path}")

        manifest = read_artifact_manifest(pretrained_path)
        kind = artifact_kind(manifest)
        if kind in {"model", "pipeline"}:
            pretrained_path = artifact_component_path(pretrained_path, manifest, "processor")
            manifest = read_artifact_manifest(pretrained_path, expected_kind="processor")
        elif kind != "processor":
            raise ValueError(f"manifest.json kind {kind!r} is not 'processor'")
        with remote_code_import_path(pretrained_path, trust_remote_code=trust_remote_code):
            processor_cls = cls._processor_class_from_manifest(manifest)
        if cls is not ProcessorMixin and processor_cls is not cls:
            raise TypeError(f"{cls.__name__}.from_pretrained cannot load artifact for {processor_cls.__name__}")
        artifact_config = Config.from_json(pretrained_path / _manifest_file(manifest, "config"))
        processor_config = artifact_config.copy()
        for key, value in ({} if config_overrides is None else dict(config_overrides)).items():
            warnings.warn(f"overriding processor config value {key!r}", stacklevel=2)
            processor_config[key] = value
        return cast(T, processor_cls._from_config(processor_config))

    def _config_for_save(self) -> Config:
        r"""
        Return the persisted runtime processor config.

        Returns:
            Config written to ``processor_config.json``.
        """

        return Config(self._processor_config())

    def _manifest_for_save(self) -> Config:
        r"""
        Build the root artifact manifest for this processor.

        Returns:
            Manifest config written to ``manifest.json``.
        """

        return build_artifact_manifest(
            kind="processor",
            entrypoint=_qualified_class_name(type(self)),
            files={"config": self.CONFIG_NAME},
        )

    def _processor_config(self) -> Mapping[str, object]:
        r"""
        Return the constructor kwargs to persist; subclasses override this.

        Returns:
            Mapping of constructor keyword arguments.

        Raises:
            NotImplementedError: Always, unless a subclass overrides it.
        """

        raise NotImplementedError

    @classmethod
    def _from_config(cls, config: Config) -> Self:
        r"""
        Reinstantiate the processor from its persisted (metadata-stripped) config.

        Args:
            config: Constructor kwargs from ``processor_config.json``.

        Returns:
            The reconstructed processor.
        """

        return cls(**dict(config))

    @classmethod
    def _processor_class_from_manifest(cls, manifest: Config) -> type[ProcessorMixin]:
        r"""
        Resolve the concrete processor class named by a saved manifest's ``entrypoint`` field.

        Args:
            manifest: Loaded artifact manifest.

        Returns:
            The processor class to instantiate.

        Raises:
            ValueError: If ``entrypoint`` is missing or not fully qualified.
        """

        return _resolve_processor_class(entrypoint(manifest))


def _qualified_class_name(cls: type[object]) -> str:
    r"""
    Return the fully qualified ``module.qualname`` string used to persist a processor class.

    Args:
        cls: Class to name.

    Returns:
        Fully qualified class name.
    """

    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_processor_class(class_name: str) -> type[ProcessorMixin]:
    r"""
    Import and return the ``ProcessorMixin`` subclass named by a fully qualified string.

    Args:
        class_name: Fully qualified ``module.qualname`` of a processor class.

    Returns:
        The resolved processor class.

    Raises:
        ValueError: If the name is not fully qualified.
        TypeError: If it resolves to something other than a ``ProcessorMixin`` subclass.
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
    if not issubclass(target, ProcessorMixin):
        raise TypeError(f"{class_name} is not an OFoldX ProcessorMixin subclass")
    return target


def _manifest_file(manifest: Config, name: str) -> str:
    r"""
    Return a named file path from a processor manifest.

    Args:
        manifest: Parsed processor artifact manifest.
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
