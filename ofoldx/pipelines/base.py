r"""
Base pipeline abstractions shared by every task pipeline.

Defines the [`Pipeline`][ofoldx.pipelines.base.Pipeline] inference boundary (model plus optional structure
processor), the [`StructureProcessor`][ofoldx.pipelines.base.StructureProcessor] protocol its processors satisfy,
and the input predicate that decides when a raw structure must first be run through a processor. The generic
``from_pretrained`` here also dispatches a plain root-level model artifact to its concrete task pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, Generic, Protocol, TypeGuard, TypeVar, cast

import torch
from torch import nn

from ..artifacts import artifact_task, read_artifact_manifest
from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, remote_code_import_path, resolve_pretrained_path
from ..models.contract import BaseInput
from ..processors.utils import StructureInput
from .auto import (
    PIPELINE_MANIFEST_NAME,
    has_pipeline_manifest,
    read_pipeline_manifest,
    resolve_pipeline_class,
    validate_pipeline_class,
)


def is_structure_input(value: object) -> TypeGuard[StructureInput]:
    r"""
    Return whether ``value`` is a raw structure input that needs a processor.

    A raw input is a parsed [`Structure`][ofoldx.data.structure.Structure] or a filesystem path (``str`` or
    [`os.PathLike`][os.PathLike]); such values must be run through a processor before the model. Already-processed model
    inputs are deliberately not enumerated here — custom model inputs remain valid as long as they implement ``to(...)``
    for device movement — so this narrows to the structure branch rather than validating every accepted input.

    Args:
        value: Candidate pipeline input to classify.

    Returns:
        ``True`` when ``value`` is a structure object or a path that needs processing, ``False`` otherwise.

    Examples:
        >>> is_structure_input("complex.cif")
        True
        >>> is_structure_input(42)
        False
    """

    return isinstance(value, (Structure, str, PathLike))


InputT = TypeVar("InputT", bound=BaseInput)
ProcessorInputT = TypeVar("ProcessorInputT", bound=BaseInput, covariant=True)
OutputT = TypeVar("OutputT")


class StructureProcessor(Protocol[ProcessorInputT]):
    r"""
    Structural typing protocol for callables that turn a structure object or path into a typed model input.

    A pipeline accepts any object matching this protocol as its ``processor``; concrete processors such as those built
    by [`AutoProcessor`][ofoldx.processors.AutoProcessor] satisfy it without explicit subclassing. The single type
    variable ``ProcessorInputT`` is the model-input dataclass the processor produces, letting a pipeline stay generic
    over its input type.

    Examples:
        >>> from ofoldx.processors import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
        >>> model_input = processor("complex.cif")  # doctest: +SKIP
    """

    def __call__(self, structure: StructureInput, /) -> ProcessorInputT:
        r"""
        Convert one structure input into a typed model input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Model input dataclass with tensors ready for batching/device movement.

        Examples:
            >>> processor("complex.cif")  # doctest: +SKIP
        """

        ...


class Pipeline(Generic[InputT, OutputT]):
    r"""
    Thin inference boundary pairing a model with an optional structure processor.

    Concrete task pipelines (design, structure prediction) subclass this to run a structure through the processor when
    needed, move the resulting batch to the model device, and normalize the model output. The model is switched to
    evaluation mode on construction. The generic ``InputT``/``OutputT`` type variables are the typed model input and the
    task output the subclass produces.

    Args:
        model: Model module used for inference; placed in ``eval`` mode on construction.
        processor: Optional processor for structure objects or paths. Already-typed model inputs do not require one.

    Examples:
        >>> from ofoldx.pipelines import DesignPipeline
        >>> pipeline = DesignPipeline.from_pretrained("oteam/design-model")  # doctest: +SKIP
        >>> pipeline("complex.cif")  # doctest: +SKIP
    """

    def __init__(self, model: nn.Module, processor: StructureProcessor[InputT] | None = None) -> None:
        self.model = model.eval()
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_pipeline_name_or_path: str | PathLike[str],
        *,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Pipeline[Any, Any]:
        r"""
        Load a pipeline from a root-level model artifact or pipeline package.

        Called on the base [`Pipeline`][ofoldx.pipelines.base.Pipeline] it resolves the concrete task class: a saved
        pipeline package is dispatched via its root ``manifest.json``, while a plain model artifact is routed to the
        task pipeline registered for its architecture. Called on a concrete subclass it instead validates that the
        package targets that subclass and delegates to the subclass loader.

        Args:
            pretrained_pipeline_name_or_path: Local pretrained directory or Hugging Face Hub repository id.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.
            **kwargs: Extra keyword arguments forwarded to the concrete pipeline loader.

        Returns:
            Loaded concrete pipeline instance.

        Raises:
            TypeError: If the package does not identify a concrete OFoldX pipeline class or the concrete class
                does not implement ``from_pretrained``.

        Examples:
            >>> from ofoldx.pipelines import Pipeline
            >>> pipeline = Pipeline.from_pretrained("oteam/design-model")  # doctest: +SKIP
            >>> pipeline("complex.cif")  # doctest: +SKIP
        """

        package_path = resolve_pretrained_path(
            pretrained_pipeline_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                PIPELINE_MANIFEST_NAME,
                "config.json",
                trust_remote_code=trust_remote_code,
            ),
        )
        if not has_pipeline_manifest(package_path):
            return _load_plain_model_pipeline(
                pretrained_pipeline_name_or_path,
                pretrained_path=package_path,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        manifest = read_pipeline_manifest(package_path)
        with remote_code_import_path(package_path, trust_remote_code=trust_remote_code):
            if cls is Pipeline:
                pipeline_cls = resolve_pipeline_class(manifest)
                if pipeline_cls is Pipeline:
                    raise TypeError("manifest.json must point to a concrete pipeline class")
                if not issubclass(pipeline_cls, Pipeline):
                    raise TypeError(f"pretrained pipeline class {pipeline_cls.__name__} is not an OFoldX Pipeline")
            else:
                validate_pipeline_class(manifest, cls)
                pipeline_cls = cls
        from_pretrained = getattr(pipeline_cls, "from_pretrained", None)
        if not callable(from_pretrained):
            raise TypeError(f"{pipeline_cls.__name__} must implement from_pretrained")
        load = cast(Callable[..., object], from_pretrained)
        loader = getattr(load, "__func__", load)
        base_loader = getattr(Pipeline.from_pretrained, "__func__", Pipeline.from_pretrained)
        if loader is base_loader:
            raise TypeError(f"{pipeline_cls.__name__} must implement from_pretrained")
        return cast(
            Pipeline[Any, Any],
            load(
                pretrained_pipeline_name_or_path,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ),
        )

    @property
    def device(self) -> torch.device:
        r"""
        Return the first parameter device, or CPU for parameter-free test doubles.

        Returns:
            Device used to move model inputs before inference.

        Examples:
            >>> from torch import nn
            >>> pipeline = Pipeline(nn.Linear(2, 2))
            >>> pipeline.device.type
            'cpu'
            >>> Pipeline(nn.Identity()).device.type
            'cpu'
        """

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_model(self, batch: InputT) -> InputT:
        r"""
        Move a typed model input onto the model device.

        Args:
            batch: Typed model input exposing ``to(device=...)``.

        Returns:
            The same input relocated to [`device`][ofoldx.pipelines.base.Pipeline.device].
        """

        return batch.to(device=self.device)

    def __call__(self, inputs: InputT, **kwargs: Any) -> OutputT:
        r"""
        Run inference for this pipeline.

        Concrete task pipelines implement the actual input preparation and output normalization.
        """

        raise NotImplementedError(f"{type(self).__name__} must implement __call__")


def _load_plain_model_pipeline(
    pretrained_model_name_or_path: str | PathLike[str],
    *,
    pretrained_path: str | PathLike[str],
    repo_type: str,
    revision: str | None,
    cache_dir: str | PathLike[str] | None,
    local_files_only: bool,
    token: str | bool | None,
    trust_remote_code: bool,
    **kwargs: Any,
) -> Pipeline[Any, Any]:
    r"""
    Load the task pipeline for a plain root-level model artifact.

    Reads the artifact config to determine its architecture, then dispatches to the design or structure-prediction
    pipeline loader registered for that architecture. Used by
    [`Pipeline.from_pretrained`][ofoldx.pipelines.base.Pipeline.from_pretrained] when the artifact has no
    pipeline ``manifest.json`` (i.e. it is a bare model rather than a saved pipeline package).

    Args:
        pretrained_model_name_or_path: Original artifact reference passed through to the concrete loader.
        pretrained_path: Resolved local directory used to read the model config.
        repo_type: Hub repository type forwarded to the concrete loader.
        revision: Optional Hub branch, tag, or commit id.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: If ``True``, load only from local files or cache.
        token: Optional Hugging Face authentication token.
        trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.
        **kwargs: Extra keyword arguments forwarded to the concrete pipeline loader.

    Returns:
        Loaded design or structure-prediction pipeline for the artifact.

    Raises:
        TypeError: If the artifact architecture is not registered for any pipeline task.
    """

    from ..models import AutoConfig
    from ..models.registry import (
        DESIGN_GENERATOR_MODELS,
        STRUCTURE_PREDICTION_MODELS,
        architecture_from_config,
    )
    from ..processors import AutoProcessor
    from .binding_affinity import BindingAffinityPipeline
    from .design import DesignPipeline
    from .side_chain_packing import SideChainPackingPipeline
    from .structure_prediction import StructurePredictionPipeline

    config = AutoConfig.from_pretrained(
        pretrained_path,
        trust_remote_code=trust_remote_code,
    )
    manifest = read_artifact_manifest(pretrained_path, expected_kind="model")
    task = artifact_task(manifest)
    architecture = architecture_from_config(config)
    if task == "binding_affinity_prediction":
        load_kwargs = dict(kwargs)
        if load_kwargs.get("processor") is None:
            load_kwargs["processor"] = _load_plain_model_processor(
                AutoProcessor,
                pretrained_path,
                manifest,
                config,
                processor_config_overrides=cast(
                    Mapping[str, object] | None,
                    load_kwargs.get("processor_config_overrides"),
                ),
                trust_remote_code=trust_remote_code,
            )
        return BindingAffinityPipeline.from_pretrained(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
    if task == "side_chain_packing":
        return SideChainPackingPipeline.from_pretrained(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    load_kwargs = dict(kwargs)
    if load_kwargs.get("processor") is None:
        load_kwargs["processor"] = _load_plain_model_processor(
            AutoProcessor,
            pretrained_path,
            manifest,
            config,
            processor_config_overrides=cast(
                Mapping[str, object] | None,
                load_kwargs.get("processor_config_overrides"),
            ),
            trust_remote_code=trust_remote_code,
        )
    if architecture in DESIGN_GENERATOR_MODELS:
        return DesignPipeline.from_pretrained(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
    if architecture in STRUCTURE_PREDICTION_MODELS:
        return StructurePredictionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
    raise TypeError(f"pretrained artifact architecture {architecture!r} is not registered for a pipeline task")


def _load_plain_model_processor(
    auto_processor: Any,
    pretrained_path: str | PathLike[str],
    manifest: Any,
    config: Any,
    *,
    processor_config_overrides: Mapping[str, object] | None,
    trust_remote_code: bool,
) -> Any:
    processor_kwargs = dict(processor_config_overrides or {})
    components = manifest.get("components", {})
    if isinstance(components, Mapping) and "processor" in components:
        return auto_processor.from_pretrained(
            pretrained_path,
            config_overrides=processor_kwargs,
            trust_remote_code=trust_remote_code,
        )
    return auto_processor.from_config(config, **processor_kwargs)
