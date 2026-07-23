r"""Persistence and device transfer for model-processor pipelines."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Generic, TypeVar, cast

import chanfig
import torch
from typing_extensions import Self

from ..artifacts import SaveableModel, save_model
from ..model import Model
from ..processor import Processor
from .generator import PipelineProcessor
from .registry import PIPELINES

_ModelT = TypeVar("_ModelT", bound=SaveableModel)
_ProcessorT = TypeVar("_ProcessorT", bound=PipelineProcessor)


class PairedPipeline(Generic[_ModelT, _ProcessorT]):
    r"""Base class for a pipeline owning one Model and Processor.

    Args:
        model: Model that implements the biological operation.
        processor: Processor that lowers public inputs for ``model``.
    """

    def __init__(self, *, model: _ModelT, processor: _ProcessorT) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Move the bound Model and Processor to one device and dtype.

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
        r"""Save the pair and optional pipeline marker, then upload when requested.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        pipeline_id = PIPELINES.id_for(type(self))
        save_model(
            directory,
            model=cast(Model, self.model),
            processor=cast(Processor, self.processor),
            max_shard_size=max_shard_size,
        )
        if pipeline_id is not None:
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
