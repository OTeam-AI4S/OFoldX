r"""
Persistence and device transfer for model-processor Pipelines.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, cast

import torch
from typing_extensions import Self

from ..artifacts import save_model
from ..hub import PROCESSOR_METADATA, load_model, load_processor, model_values, resolve_repository
from ..model import Model
from ..models.registry import MODELS
from ..modules.backend import Backend
from ..processor import Processor
from .registry import PIPELINES

_ModelT = TypeVar("_ModelT", bound=Model)
_ProcessorT = TypeVar("_ProcessorT", bound=Processor)


def _require_pair_identity(
    owner: str,
    architecture: str,
    model: Model,
    processor: Processor,
) -> None:
    expected = MODELS.lookup_model(architecture)
    if type(model) is expected.model and type(processor) is expected.processor:
        MODELS.for_pair(model, processor)
        return
    raise TypeError(
        f"{owner} requires architecture {architecture!r} "
        f"({expected.model.__name__}, {expected.processor.__name__}); "
        f"got ({type(model).__name__}, {type(processor).__name__})"
    )


class PairedPipeline(Generic[_ModelT, _ProcessorT]):
    r"""
    Base class for a Pipeline backed by one registered Model and Processor.

    Args:
        model: Model that performs the task computation.
        processor: Processor that converts user inputs into model inputs.
    """

    model_architecture: ClassVar[str]

    def __init__(self, *, model: _ModelT, processor: _ProcessorT) -> None:
        _require_pair_identity(type(self).__name__, self.model_architecture, model, processor)
        self.model = model
        self.processor = processor

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
        Load the registered pair from one repository snapshot.

        Args:
            repository: Local directory or Hugging Face Hub repository ID.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for the loaded Model and Processor.
            dtype: Floating-point dtype for the loaded Model and Processor.
            backend: Optional model operator backend.

        Returns:
            A Pipeline bound to the loaded pair.
        """

        resolved = resolve_repository(
            repository,
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
            This Pipeline after in-place placement.
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
        Save the pair and Pipeline marker, then upload when requested.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the complete saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        pipeline_id = PIPELINES.id_for(type(self))
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
