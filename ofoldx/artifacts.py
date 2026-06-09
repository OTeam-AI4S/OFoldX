"""
Shared artifact manifest helpers.

The root ``manifest.json`` describes the saved artifact as a package: its physical kind, import entrypoint,
producer, and any component paths. Runtime configs stay in their own files.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from chanfig import Config

from ._version import __version__

ARTIFACT_MANIFEST_NAME = "manifest.json"
ARTIFACT_MANIFEST_FORMAT = "biomolecular_pretrained"
ARTIFACT_MANIFEST_VERSION = 1


def normalize_task(task: str) -> str:
    """
    Return the canonical form of a task name.

    Args:
        task: Task name supplied by a manifest writer or caller.

    Returns:
        Lowercase task name with hyphens normalized to underscores.

    Raises:
        TypeError: If ``task`` is not a string.
        ValueError: If ``task`` is empty after stripping.
    """

    if not isinstance(task, str):
        raise TypeError("task must be a string")
    normalized = task.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("task must be a non-empty string")
    return normalized


def build_artifact_manifest(
    *,
    kind: str,
    entrypoint: str,
    architecture: str | None = None,
    task: str | None = None,
    components: Mapping[str, Mapping[str, object] | str] | None = None,
    files: Mapping[str, str] | None = None,
    weights: Mapping[str, str] | None = None,
) -> Config:
    """
    Build a serializable artifact manifest.

    Args:
        kind: Physical artifact kind, such as ``"model"``, ``"processor"``, or ``"pipeline"``.
        entrypoint: Fully-qualified Python class path used to load the artifact.
        architecture: Optional model architecture key.
        task: Optional task served by the artifact.
        components: Optional component mapping. String values are wrapped as ``{"path": value}``.
        files: Optional named file mapping.
        weights: Optional named weights mapping.

    Returns:
        Manifest config suitable for writing to ``manifest.json``.
    """

    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("artifact kind must be a non-empty string")
    if not isinstance(entrypoint, str) or "." not in entrypoint:
        raise ValueError("artifact entrypoint must be fully qualified")
    manifest = Config(
        {
            "format": ARTIFACT_MANIFEST_FORMAT,
            "format_version": ARTIFACT_MANIFEST_VERSION,
            "kind": kind.strip().lower(),
            "producer": {"package": entrypoint.split(".", 1)[0], "version": __version__},
            "entrypoint": entrypoint,
        }
    )
    if architecture is not None:
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("artifact architecture must be a non-empty string")
        manifest["architecture"] = architecture.lower()
    if task is not None:
        manifest["task"] = normalize_task(task)
    if components is not None:
        manifest["components"] = {
            name: {"path": value} if isinstance(value, str) else dict(value) for name, value in components.items()
        }
    if files is not None:
        manifest["files"] = dict(files)
    if weights is not None:
        manifest["weights"] = dict(weights)
    return manifest


def has_artifact_manifest(path: str | PathLike[str]) -> bool:
    """
    Return whether ``path`` directly contains ``manifest.json``.
    """

    return (Path(path) / ARTIFACT_MANIFEST_NAME).is_file()


def read_artifact_manifest(path: str | PathLike[str], *, expected_kind: str | None = None) -> Config:
    """
    Read and validate the root ``manifest.json`` under an artifact directory.

    Args:
        path: Artifact directory.
        expected_kind: Optional artifact kind to require.

    Returns:
        Parsed manifest.

    Raises:
        FileNotFoundError: If ``path`` or ``manifest.json`` is missing.
        NotADirectoryError: If ``path`` is not a directory.
        ValueError: If the manifest metadata is malformed or unsupported.
    """

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"pretrained artifact path does not exist: {artifact_path}")
    if not artifact_path.is_dir():
        raise NotADirectoryError(f"pretrained artifact path must be a directory: {artifact_path}")
    manifest_path = artifact_path / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pretrained artifact manifest does not exist: {manifest_path}")
    manifest = Config.from_json(manifest_path)
    validate_artifact_manifest(manifest, expected_kind=expected_kind)
    return manifest


def validate_artifact_manifest(manifest: Config, *, expected_kind: str | None = None) -> None:
    """
    Validate root manifest metadata.

    Args:
        manifest: Parsed artifact manifest.
        expected_kind: Optional artifact kind to require.

    Raises:
        ValueError: If required manifest metadata is missing, malformed, or unsupported.
    """

    format_name = manifest.get("format")
    if format_name != ARTIFACT_MANIFEST_FORMAT:
        raise ValueError(f"manifest.json has unsupported format {format_name!r}")
    format_version = manifest.get("format_version")
    if not isinstance(format_version, int):
        raise ValueError("manifest.json must contain an integer format_version")
    if format_version != ARTIFACT_MANIFEST_VERSION:
        raise ValueError(
            f"manifest.json format_version {format_version!r} is not supported; "
            f"expected {ARTIFACT_MANIFEST_VERSION}"
        )
    kind = manifest.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("manifest.json must contain a non-empty kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"manifest.json kind {kind!r} is not {expected_kind!r}")
    entrypoint(manifest)


def artifact_kind(manifest: Config) -> str:
    """
    Return the manifest artifact kind.

    Args:
        manifest: Parsed artifact manifest.

    Returns:
        Artifact kind string.
    """

    kind = manifest.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("manifest.json must contain a non-empty kind")
    return kind


def artifact_task(manifest: Config) -> str | None:
    """
    Return the normalized task declared by a manifest, if any.
    """

    task = manifest.get("task")
    if task is None:
        return None
    return normalize_task(task)


def artifact_component_path(path: str | PathLike[str], manifest: Config, *names: str) -> Path:
    """
    Return the first matching component directory declared by a pipeline manifest.
    """

    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("manifest.json must contain a components mapping")
    for name in names:
        component = components.get(name)
        relative: object | None
        if component is None:
            continue
        if isinstance(component, str):
            relative = component
        elif isinstance(component, Mapping):
            relative = component.get("path")
        else:
            relative = None
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"component {name!r} must contain a string path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"component {name!r} path must stay inside the artifact")
        return Path(path) / relative_path
    raise ValueError(f"manifest.json is missing component {'/'.join(names)!r}")


def entrypoint(manifest: Config) -> str:
    """
    Return a manifest's fully-qualified Python entrypoint.
    """

    value = manifest.get("entrypoint")
    if not isinstance(value, str) or "." not in value:
        raise ValueError("manifest.json must contain a fully qualified entrypoint")
    return value


def resolve_entrypoint(manifest: Config) -> object:
    """
    Import and return the Python object named by a manifest entrypoint.
    """

    class_name = entrypoint(manifest)
    module_name, _, attr_name = class_name.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"entrypoint must be fully qualified: {class_name!r}")
    module = importlib.import_module(module_name)
    target: object = module
    for part in attr_name.split("."):
        target = getattr(target, part)
    return target
