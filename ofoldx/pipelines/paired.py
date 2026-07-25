r"""
Persistence and device transfer for model-processor pipelines.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Generic, TypeVar, cast

import chanfig
import torch
from typing_extensions import Self

from ..artifacts import save_model
from ..hub import load_model, load_processor, resolve_repository
from ..model import Model
from ..processor import Processor
from .registry import PIPELINES

_ModelT = TypeVar("_ModelT", bound=Model)
_ProcessorT = TypeVar("_ProcessorT", bound=Processor)


class PairedPipeline(Generic[_ModelT, _ProcessorT]):
    r"""
    Base class for a pipeline backed by one model and processor.

    Args:
        model: Model that performs the task computation.
        processor: Processor that converts user inputs into model inputs.
    """

    def __init__(self, *, model: _ModelT, processor: _ProcessorT) -> None:
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
    ) -> Self:
        r"""
        Load the registered model-processor pair from one repository snapshot.

        Args:
            repository: Local directory or Hugging Face Hub repository ID.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for the loaded model and processor.
            dtype: Floating-point dtype for the loaded model and processor.

        Returns:
            A pipeline bound to the loaded pair.
        """

        resolved = resolve_repository(repository, revision=revision, token=token)
        processor = load_processor(resolved, device=device, dtype=dtype)
        model = load_model(resolved, device=device, dtype=dtype, token=token)
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
        Move the bound model and processor to one device and dtype.

        Args:
            device: Destination device. ``None`` preserves the current device.
            dtype: Destination floating-point dtype. ``None`` preserves the current dtype.

        Returns:
            This pipeline after in-place placement.
        """

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
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
        Save the registered pair and pipeline marker, then upload when requested.

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
        save_model(
            directory,
            model=self.model,
            processor=self.processor,
            max_shard_size=max_shard_size,
        )
        chanfig.NestedDict({"pipeline": pipeline_id}).dump(Path(directory).expanduser() / "pipeline.json")
        if repo_id is not None:
            from ..hub import upload_repository

            upload_repository(
                directory,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )
