r"""
Persistence for pipelines owning multiple model-processor pairs.

``pair_layout`` assigns one pair to the repository root and additional pairs to fixed
subdirectories. Only the root carries ``pipeline.json``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple, cast

import chanfig
import torch
from typing_extensions import Self

from ..artifacts import save_model
from ..hub import (
    PROCESSOR_METADATA,
    ResolvedRepository,
    hydrate_weights,
    load_registered_model,
    load_registered_processor,
    model_values,
    resolve_repository,
)
from ..model import Model
from ..processor import Processor
from .paired import PairedPipeline, _require_pair_identity
from .registry import PIPELINES


class CompositePair(NamedTuple):
    r"""
    One pair in a composite pipeline's repository layout.

    Attributes:
        subdirectory: Repository-relative location; ``""`` selects the root.
        architecture: Expected registered Model architecture.
        model_attribute: Model or wrapped pipeline constructor attribute.
        processor_attribute: Processor attribute, or ``None`` for a wrapped pair.
        wrap: Optional pipeline class used to wrap a loaded pair.
        required: Whether the pair must be present.
    """

    subdirectory: str
    architecture: str
    model_attribute: str
    processor_attribute: str | None
    wrap: type | None = None
    required: bool = True


def _pair_directory(root: Path, subdirectory: str) -> Path:
    posix = PurePosixPath(subdirectory)
    windows = PureWindowsPath(subdirectory)
    if posix.anchor or windows.anchor or ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"composite pair path must stay within the repository: {subdirectory!r}")
    return root / subdirectory


class CompositePipeline:
    r"""
    Base class for pipelines with a fixed multi-pair repository layout.
    """

    pair_layout: tuple[CompositePair, ...]

    def _composite_pair_source(self, pair: CompositePair) -> tuple[Model, Processor] | None:
        r"""
        Return the ``(model, processor)`` a layout slot serializes, or ``None`` when absent.
        """

        bound = getattr(self, pair.model_attribute)
        if pair.wrap is not None:
            if bound is None:
                return None
            if type(bound) is not pair.wrap:
                location = pair.subdirectory or "root"
                raise TypeError(
                    f"{type(self).__name__} {location} requires {pair.wrap.__name__}; " f"got {type(bound).__name__}"
                )
            wrapped = cast(PairedPipeline[Model, Processor], bound)
            model = wrapped.model
            processor = wrapped.processor
            _require_pair_identity(
                f"{type(self).__name__} {pair.subdirectory or 'root'}",
                pair.architecture,
                model,
                processor,
            )
            return model, processor
        processor_attribute = pair.processor_attribute
        assert processor_attribute is not None
        processor = getattr(self, processor_attribute)
        if (bound is None) != (processor is None):
            location = pair.subdirectory or "root"
            raise ValueError(f"{location} model and processor must either both be present or both be absent")
        if bound is None:
            return None
        _require_pair_identity(
            f"{type(self).__name__} {pair.subdirectory or 'root'}",
            pair.architecture,
            bound,
            processor,
        )
        return bound, processor

    def _validate_components(self) -> None:
        for pair in self.pair_layout:
            source = self._composite_pair_source(pair)
            if source is None and pair.required:
                location = pair.subdirectory or "root"
                raise ValueError(f"{type(self).__name__} requires the {location} pair")

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Move every present owned pair or wrapped pipeline to one device and dtype.

        Args:
            device: Destination device. ``None`` preserves each component's current device.
            dtype: Destination floating-point dtype. ``None`` preserves each component's dtype.

        Returns:
            This pipeline after in-place placement.
        """

        for pair in self.pair_layout:
            source = self._composite_pair_source(pair)
            if source is None:
                continue
            if pair.wrap is not None:
                getattr(self, pair.model_attribute).to(device=device, dtype=dtype)
                continue
            model, processor = source
            model.to(device=device, dtype=dtype)
            processor.to(device=device, dtype=dtype)
        return self

    def save(
        self,
        directory: str | PathLike[str],
        *,
        max_shard_size: int | str = "5GB",
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        r"""
        Save every pair, write the root marker, and optionally upload the repository.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        pipeline_id = PIPELINES.id_for(type(self))
        if pipeline_id is None:
            raise ValueError(f"{type(self).__name__} is not a registered concrete Pipeline; it cannot be saved")

        destination = Path(directory).expanduser()
        self._validate_components()

        # Resolve every slot's source first so a missing required pair aborts the save
        # before the destination is mutated.
        plan: list[tuple[CompositePair, Path, tuple[Model, Processor] | None]] = []
        for pair in self.pair_layout:
            pair_directory = _pair_directory(destination, pair.subdirectory)
            source = self._composite_pair_source(pair)
            plan.append((pair, pair_directory, source))

        for pair, pair_directory, source in plan:
            if source is None:
                if pair.subdirectory:
                    stale = pair_directory
                    for filename in (
                        "model.json",
                        "processor.json",
                        "pipeline.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                    ):
                        (stale / filename).unlink(missing_ok=True)
                    for shard in stale.glob("model-*-of-*.safetensors"):
                        shard.unlink()
                continue
            model, processor = source
            save_model(
                pair_directory,
                model=model,
                processor=processor,
                max_shard_size=max_shard_size,
            )

        chanfig.NestedDict({"pipeline": pipeline_id}).dump(destination / "pipeline.json")

        if repo_id is not None:
            from ..hub import upload_repository

            upload_repository(
                destination,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )

    @classmethod
    def from_repository(
        cls,
        repository: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Load every pair from one resolved repository snapshot.

        Missing optional pairs are skipped; missing required pairs raise ``ValueError``.

        Args:
            repository: Local directory or Hub repository ID.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for loaded models and processors.
            dtype: Floating-point dtype for loaded models and processors.

        Returns:
            A pipeline assembled from the repository's declared pairs.
        """

        metadata_patterns = tuple(
            str(PurePosixPath(pair.subdirectory) / filename)
            for pair in cls.pair_layout
            for filename in PROCESSOR_METADATA
        )
        resolved = resolve_repository(
            repository,
            revision=revision,
            allow_patterns=metadata_patterns,
            token=token,
        )
        components: list[tuple[CompositePair, Model, Processor]] = []
        for pair in cls.pair_layout:
            subrepository = ResolvedRepository(_pair_directory(resolved.path, pair.subdirectory))
            if not (subrepository.path / "model.json").exists():
                if pair.required:
                    location = pair.subdirectory or "root"
                    raise ValueError(f"repository is missing the {location} pair required by {cls.__name__}")
                continue
            if not (subrepository.path / "processor.json").exists():
                location = pair.subdirectory or "root"
                raise ValueError(f"repository is missing processor.json for the {location} pair")
            registration, values = model_values(subrepository.path)
            if registration.id != pair.architecture:
                raise ValueError(
                    f"{cls.__name__} {pair.subdirectory or 'root'} requires architecture {pair.architecture!r}; "
                    f"repository declares {registration.id!r}"
                )
            config = registration.model_config(values)
            with torch.device("meta"):
                model = registration.model(config)
            processor = load_registered_processor(
                subrepository,
                registration,
                device=device,
                dtype=dtype,
            )
            components.append((pair, model, processor))

        arguments: dict[str, object] = {}
        for pair, model, processor in components:
            subrepository = hydrate_weights(
                resolved,
                subdirectory=pair.subdirectory,
                token=token,
            )
            model = load_registered_model(
                subrepository,
                model,
                device=device,
                dtype=dtype,
            )
            if pair.wrap is not None:
                arguments[pair.model_attribute] = pair.wrap(model=model, processor=processor)
            else:
                arguments[pair.model_attribute] = model
                processor_attribute = pair.processor_attribute
                assert processor_attribute is not None
                arguments[processor_attribute] = processor
        return cls(**arguments)
