r"""
Persistence for Pipelines owning multiple Model-Processor pairs.

``pair_layout`` assigns one pair to the repository root and additional pairs to
fixed subdirectories. Only the root carries ``pipeline.json``.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import NamedTuple, cast

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
from ..modules.backend import Backend
from ..processor import Processor
from .paired import PairedPipeline, _require_pair_identity
from .registry import PIPELINES


class CompositePair(NamedTuple):
    r"""
    One pair in a composite Pipeline's repository layout.

    Attributes:
        subdirectory: Repository-relative location; ``""`` selects the root.
        architecture: Expected registered Model architecture.
        model_attribute: Model or wrapped Pipeline constructor attribute.
        processor_attribute: Processor attribute, or ``None`` for a wrapped pair.
        wrap: Optional Pipeline class used to wrap a loaded pair.
    """

    subdirectory: str
    architecture: str
    model_attribute: str
    processor_attribute: str | None
    wrap: type | None = None


class CompositePipeline:
    r"""
    Base class for Pipelines with a fixed multi-pair repository layout.
    """

    pair_layout: tuple[CompositePair, ...]

    def _composite_pair_source(self, pair: CompositePair) -> tuple[Model, Processor]:
        r"""
        Return the pair serialized by one fixed layout slot.
        """

        bound = getattr(self, pair.model_attribute)
        if pair.wrap is not None:
            if type(bound) is not pair.wrap:
                location = pair.subdirectory or "root"
                raise TypeError(
                    f"{type(self).__name__} {location} requires {pair.wrap.__name__}; got {type(bound).__name__}"
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

        processor_attribute = cast(str, pair.processor_attribute)
        processor = getattr(self, processor_attribute)
        _require_pair_identity(
            f"{type(self).__name__} {pair.subdirectory or 'root'}",
            pair.architecture,
            bound,
            processor,
        )
        return bound, processor

    def _validate_components(self) -> None:
        for pair in self.pair_layout:
            self._composite_pair_source(pair)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Move every owned pair or wrapped Pipeline to one device and dtype.

        Args:
            device: Destination device. ``None`` preserves each component's current device.
            dtype: Destination floating-point dtype. ``None`` preserves each component's dtype.

        Returns:
            This Pipeline after in-place placement.
        """

        for pair in self.pair_layout:
            if pair.wrap is not None:
                getattr(self, pair.model_attribute).to(device=device, dtype=dtype)
                continue
            model = getattr(self, pair.model_attribute)
            processor = getattr(self, cast(str, pair.processor_attribute))
            model.to(device=device, dtype=dtype)
            processor.to(device=device, dtype=dtype)
        return self

    def set_backend(self, backend: Backend) -> Self:
        r"""Select one operator backend for every owned Model."""

        for pair in self.pair_layout:
            getattr(self, pair.model_attribute).set_backend(backend)
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
        Save every pair and the root marker, then upload when requested.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the complete saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        pipeline_id = PIPELINES.id_for(type(self))
        destination = Path(directory).expanduser()

        plan = [(destination / pair.subdirectory, self._composite_pair_source(pair)) for pair in self.pair_layout]
        marker = destination / "pipeline.json"
        marker.unlink(missing_ok=True)
        for pair_directory, (model, processor) in plan:
            save_model(
                pair_directory,
                model=model,
                processor=processor,
                max_shard_size=max_shard_size,
            )

        marker.write_text(
            json.dumps({"pipeline": pipeline_id}, indent=2) + "\n",
            encoding="utf-8",
        )

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
        backend: Backend | None = None,
    ) -> Self:
        r"""
        Load every pair from one repository snapshot.

        Args:
            repository: Local directory or Hugging Face Hub repository ID.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for loaded Models and Processors.
            dtype: Floating-point dtype for loaded Models and Processors.
            backend: Optional model operator backend applied to every component.

        Returns:
            A Pipeline assembled from the repository's declared pairs.
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
            subrepository = ResolvedRepository(resolved.path / pair.subdirectory)
            registration, values = model_values(subrepository.path)
            if registration.id != pair.architecture:
                raise ValueError(
                    f"{cls.__name__} {pair.subdirectory or 'root'} requires architecture "
                    f"{pair.architecture!r}; repository declares {registration.id!r}"
                )
            config = registration.model.config_class(values)
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
            if backend is not None:
                model.set_backend(backend)
            if pair.wrap is not None:
                arguments[pair.model_attribute] = pair.wrap(model=model, processor=processor)
            else:
                arguments[pair.model_attribute] = model
                processor_attribute = cast(str, pair.processor_attribute)
                arguments[processor_attribute] = processor
        return cls(**arguments)
