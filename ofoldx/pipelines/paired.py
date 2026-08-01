# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Persistence and device transfer for Model-Processor pairs and Pipelines.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar, cast

import torch

from ..artifacts import save_model
from ..hub import PROCESSOR_METADATA, ResolvedRepository, load_model, load_processor, model_values, resolve_repository
from ..model import Model
from ..models.registry import MODELS
from ..modules.backend import Backend
from ..processor import Processor
from .base import Pipeline
from .registry import PIPELINES

_ModelT = TypeVar("_ModelT", bound=Model)
_ProcessorT = TypeVar("_ProcessorT", bound=Processor)


def _require_pair_identity(
    owner: str,
    architecture: str,
    model: Model,
    processor: Processor,
) -> None:
    try:
        registration = MODELS.for_pair(model, processor)
    except KeyError:
        raise TypeError(
            f"{owner} cannot save the unregistered Model/Processor pair "
            f"({type(model).__name__}, {type(processor).__name__}); "
            f"register the exact classes under an architecture ID and set {owner}.model_architecture to that ID"
        ) from None
    if registration.id == architecture:
        return
    raise TypeError(
        f"{owner} requires architecture {architecture!r}, "
        f"but its Model/Processor pair is registered as {registration.id!r}"
    )


class PairedComponent(Generic[_ModelT, _ProcessorT]):
    r"""
    Independently loadable component backed by one registered Model and Processor.

    Args:
        model: Model that performs the task computation.
        processor: Processor that converts user inputs into model inputs.
        source: Resolved repository snapshot that supplied the pair, or ``None``
            for a pair assembled directly from in-memory objects.

    Notes:
        Construction accepts subclasses of the registered classes. Saving requires
        the exact Model and Processor classes to be registered for this component's
        ``model_architecture`` so their concrete types can be reconstructed. For
        custom subclasses, register a new architecture and set ``model_architecture``
        to its ID.
    """

    model_architecture: ClassVar[str]
    _source: ResolvedRepository | None

    def __init__(
        self,
        *,
        model: _ModelT,
        processor: _ProcessorT,
        source: ResolvedRepository | None = None,
    ) -> None:
        expected = MODELS.lookup_model(self.model_architecture)
        if not isinstance(model, expected.model) or not isinstance(processor, expected.processor):
            raise TypeError(
                f"{type(self).__name__} requires architecture {self.model_architecture!r} "
                f"({expected.model.__name__}, {expected.processor.__name__}); "
                f"got ({type(model).__name__}, {type(processor).__name__})"
            )
        processor.validate_model_input_layout(model)
        self.model = model
        self.processor = processor
        self._source = source

    @property
    def source(self) -> ResolvedRepository | None:
        r"""Return the immutable repository snapshot that supplied this pair."""

        return self._source

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Self:
        r"""
        Load the registered pair from one repository snapshot.

        Args:
            pretrained_model_name_or_path: Local directory or Hugging Face Hub repository ID.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for the loaded Model and Processor.
            dtype: Floating-point dtype for the loaded Model and Processor.
            backend: Optional model operator backend.

        Returns:
            A component bound to the loaded pair.
        """

        resolved = resolve_repository(
            pretrained_model_name_or_path,
            revision=revision,
            allow_patterns=PROCESSOR_METADATA,
            token=token,
        )
        registration, _ = model_values(resolved.path)
        if registration.id != cls.model_architecture:
            raise ValueError(
                f"{cls.__name__} requires architecture {cls.model_architecture!r}; "
                f"repository declares {registration.id!r}"
            )
        processor = load_processor(resolved, device=device, dtype=dtype)
        model = load_model(resolved, device=device, dtype=dtype, token=token)
        if backend is not None:
            model.set_backend(backend)
        return cls(
            model=cast(_ModelT, model),
            processor=cast(_ProcessorT, processor),
            source=resolved,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Move the bound Model and Processor to one device and dtype.

        Args:
            device: Destination device. ``None`` preserves the current device.
            dtype: Destination floating-point dtype. ``None`` preserves the current dtype.

        Returns:
            This component after in-place placement.
        """

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
        return self

    def set_backend(self, backend: Backend) -> Self:
        r"""Select the operator backend used by the bound Model."""

        self.model.set_backend(backend)
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
        Save the pair, then upload when requested.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the complete saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        _require_pair_identity(
            type(self).__name__,
            self.model_architecture,
            self.model,
            self.processor,
        )
        destination = Path(directory).expanduser()
        save_model(
            destination,
            model=self.model,
            processor=self.processor,
            max_shard_size=max_shard_size,
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


class PairedPipeline(PairedComponent[_ModelT, _ProcessorT], Pipeline):
    r"""Task Pipeline backed by one registered Model and Processor.

    Construction places the Model in evaluation mode, as repository loading does.
    Evaluation mode does not disable gradients; callers may explicitly change the
    Model's mode after construction.
    """

    def __init__(
        self,
        *,
        model: _ModelT,
        processor: _ProcessorT,
        source: ResolvedRepository | None = None,
    ) -> None:
        super().__init__(model=model, processor=processor, source=source)
        self.model.eval()

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
        r"""Save the pair and Pipeline marker, then upload when requested."""

        pipeline_id = PIPELINES.id_for(type(self))
        destination = Path(directory).expanduser()
        super().save(
            destination,
            max_shard_size=max_shard_size,
        )
        (destination / "pipeline.json").write_text(
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
