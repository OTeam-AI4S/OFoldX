r"""
Auto pipeline selection and pipeline-package artifact helpers.

A pipeline package is a directory that carries a ``manifest.json`` file naming the concrete pipeline class, declaring
the task it serves, and mapping component names (``generator``/``predictor``, ``processor``) to subdirectories. The
helpers here build, read, and validate that manifest, resolve the pipeline class it points to, and derive the on-disk
names of weight and component files, so loaders can materialize a pipeline without hard-coding its layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from chanfig import Config

from ..artifacts import (
    ARTIFACT_MANIFEST_NAME,
    artifact_kind,
    artifact_task,
    build_artifact_manifest,
    entrypoint,
    normalize_task,
    read_artifact_manifest,
    resolve_entrypoint,
)

PIPELINE_MANIFEST_NAME = ARTIFACT_MANIFEST_NAME


def has_pipeline_manifest(path: str | PathLike[str]) -> bool:
    r"""
    Return whether a directory is a pipeline package.

    Args:
        path: Candidate pretrained artifact directory.

    Returns:
        ``True`` when ``manifest.json`` exists directly under ``path`` and declares ``kind: "pipeline"``.

    Examples:
        >>> has_pipeline_manifest("/nonexistent-artifact-directory")
        False
    """

    try:
        return artifact_kind(read_artifact_manifest(path)) == "pipeline"
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return False


def build_pipeline_manifest(
    class_name: str,
    components: Mapping[str, str],
    *,
    task: str,
    architecture: str | None = None,
) -> Config:
    r"""
    Return a pipeline-package manifest config.

    The manifest records the concrete pipeline ``entrypoint``, the package ``task``, the producer version, and a
    ``components`` mapping whose values wrap each relative directory in a ``{"path": ...}`` entry so component metadata
    can grow without breaking the schema.

    Args:
        class_name: Fully-qualified pipeline class name.
        components: Mapping from component name to relative directory inside the package.
        task: Task served by this package, such as ``"structure_prediction"`` or ``"design"``.
        architecture: Optional model architecture served by the pipeline package.

    Returns:
        Serializable pipeline manifest config.

    Examples:
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {"generator": "generator"},
        ...     task="design",
        ... )
        >>> manifest["entrypoint"]
        'ofoldx.pipelines.design.DesignPipeline'
        >>> manifest["task"]
        'design'
        >>> manifest["components"]["generator"]["path"]
        'generator'
    """

    return build_artifact_manifest(
        kind="pipeline",
        entrypoint=class_name,
        architecture=architecture,
        task=task,
        components={name: {"kind": "component", "path": path} for name, path in components.items()},
    )


def component_architecture(component: object) -> str | None:
    r"""
    Return a component's model architecture when it exposes an OFoldX config.
    """

    config = getattr(component, "config", None)
    if config is None:
        return None
    from ..processors.registry import architecture_from_config

    try:
        return architecture_from_config(config)
    except (TypeError, ValueError, AttributeError):
        return None


def read_pipeline_manifest(path: str | PathLike[str]) -> Config:
    r"""
    Read ``manifest.json`` from a pipeline package directory.

    Args:
        path: Pipeline package directory.

    Returns:
        Parsed pipeline manifest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        NotADirectoryError: If ``path`` is not a directory.
        ValueError: If the manifest metadata is malformed, unsupported, or not a pipeline.

    Examples:
        >>> manifest = read_pipeline_manifest("pretrained/design-pipeline")  # doctest: +SKIP
        >>> manifest["entrypoint"]  # doctest: +SKIP
        'ofoldx.pipelines.design.DesignPipeline'
    """

    manifest = read_artifact_manifest(path, expected_kind="pipeline")
    pipeline_task(manifest)
    return manifest


def pipeline_task(manifest: Config) -> str:
    r"""
    Return the task declared by a pipeline manifest.

    Args:
        manifest: Parsed pipeline manifest.

    Returns:
        Normalized task name.

    Raises:
        ValueError: If the task field is missing or malformed.

    Examples:
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.structure_prediction.StructurePredictionPipeline",
        ...     {},
        ...     task="structure-prediction",
        ... )
        >>> pipeline_task(manifest)
        'structure_prediction'
    """

    task = artifact_task(manifest)
    if task is None:
        raise ValueError("manifest.json must contain a task")
    return task


def validate_pipeline_task(manifest: Config, expected_task: str) -> None:
    r"""
    Validate that a manifest belongs to the requested task.

    Args:
        manifest: Parsed pipeline manifest.
        expected_task: Task expected by the concrete pipeline loader.

    Raises:
        TypeError: If the manifest declares a different task.

    Examples:
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {},
        ...     task="design",
        ... )
        >>> validate_pipeline_task(manifest, "design") is None
        True
        >>> validate_pipeline_task(manifest, "structure_prediction")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: pretrained pipeline task ... cannot be loaded as ...
    """

    task = pipeline_task(manifest)
    expected = normalize_task(expected_task)
    if task != expected:
        raise TypeError(f"pretrained pipeline task {task!r} cannot be loaded as {expected!r}")


def validate_pipeline_class(manifest: Config, pipeline_cls: type[object]) -> None:
    r"""
    Validate that a manifest belongs to the requested pipeline class.

    Args:
        manifest: Parsed pipeline manifest.
        pipeline_cls: Concrete pipeline class requested by the caller.

    Raises:
        TypeError: If the manifest names a different pipeline class.

    Examples:
        >>> from ofoldx.pipelines.design import DesignPipeline
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {},
        ...     task="design",
        ... )
        >>> validate_pipeline_class(manifest, DesignPipeline) is None
        True
        >>> other = build_pipeline_manifest(
        ...     "ofoldx.pipelines.structure_prediction.StructurePredictionPipeline",
        ...     {},
        ...     task="structure_prediction",
        ... )
        >>> validate_pipeline_class(other, DesignPipeline)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: pretrained pipeline class ... cannot be loaded as ...
    """

    expected = f"{pipeline_cls.__module__}.{pipeline_cls.__qualname__}"
    class_name = pipeline_entrypoint(manifest)
    if class_name != expected:
        raise TypeError(f"pretrained pipeline class {class_name!r} cannot be loaded as {expected!r}")


def pipeline_entrypoint(manifest: Config) -> str:
    r"""
    Return the fully-qualified pipeline entrypoint from a manifest.

    Args:
        manifest: Parsed pipeline manifest.

    Returns:
        Fully-qualified pipeline entrypoint.

    Raises:
        ValueError: If the manifest does not contain a fully-qualified ``entrypoint``.

    Examples:
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {},
        ...     task="design",
        ... )
        >>> pipeline_entrypoint(manifest)
        'ofoldx.pipelines.design.DesignPipeline'
    """

    return entrypoint(manifest)


def resolve_pipeline_class(manifest: Config) -> type[object]:
    r"""
    Resolve the concrete pipeline class stored in a manifest.

    Args:
        manifest: Parsed pipeline manifest.

    Returns:
        Resolved pipeline class object.

    Raises:
        ValueError: If the class name is malformed.
        TypeError: If the class name resolves to a non-class object.

    Examples:
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {},
        ...     task="design",
        ... )
        >>> resolve_pipeline_class(manifest).__name__
        'DesignPipeline'
    """

    class_name = pipeline_entrypoint(manifest)
    target = resolve_entrypoint(manifest)
    if not isinstance(target, type):
        raise TypeError(f"entrypoint does not resolve to a class: {class_name!r}")
    return target


def component_path(package_path: Path, manifest: Config, name: str) -> Path:
    r"""
    Return a named component directory from a manifest.

    Args:
        package_path: Pipeline package directory.
        manifest: Parsed pipeline manifest.
        name: Component name to resolve.

    Returns:
        Component directory path inside ``package_path``.

    Raises:
        ValueError: If the component mapping is missing, malformed, or points outside the package.

    Examples:
        >>> from pathlib import Path
        >>> manifest = build_pipeline_manifest(
        ...     "ofoldx.pipelines.design.DesignPipeline",
        ...     {"generator": "generator"},
        ...     task="design",
        ... )
        >>> component_path(Path("/tmp/pkg"), manifest, "generator")
        PosixPath('/tmp/pkg/generator')
    """

    components = manifest.get("components")
    if not isinstance(components, Config):
        raise ValueError("manifest.json must contain a components mapping")
    component = components.get(name)
    if not isinstance(component, Config):
        raise ValueError(f"manifest.json is missing component {name!r}")
    relative_path = component.get("path")
    if not isinstance(relative_path, str):
        raise ValueError(f"component {name!r} must contain a string path")
    component_relative_path = Path(relative_path)
    if component_relative_path.is_absolute() or ".." in component_relative_path.parts:
        raise ValueError(f"component {name!r} path must stay inside the pipeline package")
    return package_path / component_relative_path


def model_weights_name(variant: str | None) -> str:
    r"""
    Return the model weight filename for an optional variant.

    Args:
        variant: Optional variant suffix.

    Returns:
        ``model.safetensors`` or ``model.<variant>.safetensors``.

    Examples:
        >>> model_weights_name(None)
        'model.safetensors'
        >>> model_weights_name("fp16")
        'model.fp16.safetensors'
    """

    return "model.safetensors" if variant is None else f"model.{variant}.safetensors"


def save_component(component: object, save_directory: Path, name: str, **kwargs: object) -> None:
    r"""
    Save a pipeline component that implements the OFoldX pretrained protocol.

    Args:
        component: Component object exposing ``save_pretrained``.
        save_directory: Destination directory.
        name: Component name used in error messages.
        **kwargs: Extra keyword arguments forwarded to ``save_pretrained``.

    Raises:
        TypeError: If the component does not implement ``save_pretrained``.

    Examples:
        >>> from pathlib import Path
        >>> save_component(model, Path("pipeline/generator"), "generator")  # doctest: +SKIP
    """

    save_pretrained = getattr(component, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError(f"pipeline component {name!r} does not implement save_pretrained")
    save_pretrained(save_directory, **kwargs)
