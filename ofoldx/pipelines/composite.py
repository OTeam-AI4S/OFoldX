r"""
Persistence for Pipelines owning multiple Model-Processor pairs.

``pair_layout`` assigns one pair to the repository root and additional pairs to
fixed subdirectories. Only the root carries ``pipeline.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
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
from .paired import PairedComponent, _require_pair_identity
from .registry import PIPELINES


class CompositePair(NamedTuple):
    r"""
    One pair in a composite Pipeline's repository layout.

    Attributes:
        role: Stable semantic role used to select this component.
        subdirectory: Repository-relative location; ``""`` selects the root.
        component: Concrete component class wrapping the Model-Processor pair.
    """

    role: str
    subdirectory: str
    component: type[PairedComponent]


class CompositePipeline:
    r"""
    Base class for Pipelines with a fixed multi-pair repository layout.

    ``source`` retains the resolved repository snapshot for loaded composites and
    is ``None`` for composites assembled directly from in-memory components.

    Concrete constructors provide components keyed by the stable roles declared in
    ``pair_layout``.
    """

    pair_layout: tuple[CompositePair, ...]
    _source: ResolvedRepository | None

    def __init__(self, components: Mapping[str, PairedComponent]) -> None:
        self._components = dict(components)
        self._source = None
        self._validate_component_layout()

    @property
    def components(self) -> Mapping[str, PairedComponent]:
        r"""Return concrete components keyed by their stable semantic role."""

        return self._components

    @property
    def source(self) -> ResolvedRepository | None:
        r"""Return the immutable repository snapshot that supplied this composite."""

        return self._source

    def _composite_pair_source(self, pair: CompositePair) -> tuple[Model, Processor]:
        r"""
        Return the pair serialized by one fixed layout slot.
        """

        component = self._components[pair.role]
        if type(component) is not pair.component:
            location = pair.subdirectory or "root"
            raise TypeError(
                f"{type(self).__name__} {location} requires {pair.component.__name__}; "
                f"got {type(component).__name__}"
            )
        model = component.model
        processor = component.processor
        _require_pair_identity(
            f"{type(self).__name__} {pair.subdirectory or 'root'}",
            pair.component.model_architecture,
            model,
            processor,
        )
        return model, processor

    def _validate_component_layout(self) -> None:
        expected_roles = {pair.role for pair in self.pair_layout}
        if self._components.keys() != expected_roles:
            raise TypeError(
                f"{type(self).__name__} components must be {sorted(expected_roles)!r}; "
                f"got {sorted(self._components)!r}"
            )
        for pair in self.pair_layout:
            self._composite_pair_source(pair)

    @classmethod
    def _pair_for_role(cls, role: str) -> CompositePair:
        pairs = tuple(pair for pair in cls.pair_layout if pair.role == role)
        if len(pairs) != 1:
            raise ValueError(f"{cls.__name__} has no unique component role {role!r}")
        return pairs[0]

    @classmethod
    def _load_pair(
        cls,
        resolved: ResolvedRepository,
        pair: CompositePair,
        *,
        token: str | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
        backend: Backend | None,
    ) -> tuple[Model, Processor, ResolvedRepository]:
        subrepository = ResolvedRepository(
            resolved.path / pair.subdirectory,
            resolved.repository,
            resolved.commit,
            "/".join(part for part in (resolved.subdirectory, pair.subdirectory) if part),
        )
        registration, values = model_values(subrepository.path)
        architecture = pair.component.model_architecture
        if registration.id != architecture:
            raise ValueError(
                f"{cls.__name__} component {pair.role!r} requires architecture "
                f"{architecture!r}; repository declares {registration.id!r}"
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
        return model, processor, subrepository

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

        for component in self._components.values():
            component.to(device=device, dtype=dtype)
        return self

    def set_backend(self, backend: Backend) -> Self:
        r"""Select one operator backend for every owned Model."""

        for component in self._components.values():
            component.set_backend(backend)
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

        components: dict[str, PairedComponent] = {}
        for pair in cls.pair_layout:
            model, processor, subrepository = cls._load_pair(
                resolved,
                pair,
                token=token,
                device=device,
                dtype=dtype,
                backend=backend,
            )
            components[pair.role] = pair.component(
                model=model,
                processor=processor,
                source=subrepository,
            )
        pipeline = cast(Callable[..., Self], cls)(**components)
        pipeline._source = resolved
        return pipeline

    @classmethod
    def component_from_repository(
        cls,
        repository: str | PathLike[str],
        *,
        role: str,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> PairedComponent[Model, Processor]:
        r"""
        Load one independently deployable component by its semantic role.

        Args:
            repository: Local directory or Hugging Face Hub repository ID.
            role: Stable component role declared by ``pair_layout``.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for the loaded Model and Processor.
            dtype: Floating-point dtype for the loaded Model and Processor.
            backend: Optional model operator backend.

        Returns:
            The concrete paired component declared for ``role``.

        Raises:
            ValueError: If the role is unknown.
        """

        pair = cls._pair_for_role(role)
        metadata_patterns = tuple(str(PurePosixPath(pair.subdirectory) / filename) for filename in PROCESSOR_METADATA)
        resolved = resolve_repository(
            repository,
            revision=revision,
            allow_patterns=metadata_patterns,
            token=token,
        )
        model, processor, source = cls._load_pair(
            resolved,
            pair,
            token=token,
            device=device,
            dtype=dtype,
            backend=backend,
        )
        return cast(
            PairedComponent[Model, Processor],
            pair.component(model=model, processor=processor, source=source),
        )
