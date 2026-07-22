r"""Persistence and device transfer for model-processor pipelines."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Generic, TypeVar

import chanfig
import torch
from typing_extensions import Self

from ..artifacts import SaveableModel, save_model
from .generator import PipelineProcessor
from .registry import PIPELINES

_ModelT = TypeVar("_ModelT", bound=SaveableModel)
_ProcessorT = TypeVar("_ProcessorT", bound=PipelineProcessor)


class PairedPipeline(Generic[_ModelT, _ProcessorT]):
    r"""Base class for a pipeline owning one model and processor."""

    def __init__(self, *, model: _ModelT, processor: _ProcessorT) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
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
        r"""Save the pair and optional pipeline marker, then upload when ``repo_id`` is set."""

        pipeline_id = PIPELINES.id_for(type(self))
        save_model(
            directory,
            model=self.model,
            processor=self.processor,
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
