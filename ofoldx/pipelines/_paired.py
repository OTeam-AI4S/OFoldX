r"""Shared implementation for direct Model–Processor pair Pipelines."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import chanfig
import torch
from torch import nn
from typing_extensions import Self

from ..artifacts import save_model
from .registry import PIPELINES


class _PairedPipeline:
    r"""Shared implementation for a direct Model–Processor pair."""

    def __init__(self, *, model: nn.Module, processor: object) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)  # type: ignore[attr-defined]
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
        r"""Write the paired repository locally, and optionally publish it to the Hub.

        Saving is local-only by default. When ``repo_id`` is given the fully written
        repository — including its ``pipeline.json`` marker — is uploaded via the same
        [`ofoldx.hub`][ofoldx.hub] uploader ``save_model`` uses; nothing is published
        without an explicit ``repo_id``.
        """

        save_model(
            directory,
            model=self.model,
            processor=self.processor,
            max_shard_size=max_shard_size,
        )
        pipeline_id = PIPELINES.id_for(type(self))
        if pipeline_id is not None:
            chanfig.NestedDict({"pipeline": pipeline_id}).dump(Path(directory).expanduser() / "pipeline.json")
        if repo_id is not None:
            from ..hub import _upload_repository

            _upload_repository(
                directory,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )
