from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import ofoldx.api.tasks as tasks
from ofoldx import pipeline
from ofoldx.hub import ResolvedRepository
from ofoldx.pipelines.registry import PIPELINES, PipelineExecutionStep


class RepositoryPipeline:
    def fold(self) -> None:
        pass

    def design(self) -> None:
        pass

    @classmethod
    def from_repository(
        cls,
        repository: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: str | None = None,
    ) -> tuple[object, ...]:
        return repository, revision, token, device, dtype, backend


PIPELINES.register(
    "test_repository_pipeline",
    pipeline=RepositoryPipeline,
    executions={"fold": (PipelineExecutionStep("execute", "pipeline", RepositoryPipeline.fold),)},
)


def _write_marker(directory: Path) -> None:
    (directory / "pipeline.json").write_text(
        '{"pipeline":"test_repository_pipeline"}',
        encoding="utf-8",
    )


class TestPipeline:
    def test_local_repository(self, tmp_path: Path) -> None:
        _write_marker(tmp_path)

        loaded = pipeline(
            "fold",
            model=tmp_path,
            token="token",
            device="cpu",
            dtype=torch.float64,
            backend="eager",
        )

        assert loaded == (tmp_path, None, "token", "cpu", torch.float64, "eager")

    def test_hub_revision(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_marker(tmp_path)

        def resolve(repository: object, *, revision: str | None = None, **kwargs: object) -> ResolvedRepository:
            assert repository == "owner/model"
            assert revision == "main"
            del kwargs
            return ResolvedRepository(tmp_path, "owner/model", "commit")

        monkeypatch.setattr(tasks, "resolve_repository", resolve)

        loaded = pipeline("fold", model="owner/model", revision="main")

        assert loaded[:2] == ("owner/model", "commit")

    def test_capability(self, tmp_path: Path) -> None:
        _write_marker(tmp_path)

        with pytest.raises(ValueError, match="does not support"):
            pipeline("design", model=tmp_path)

    def test_task(self, tmp_path: Path) -> None:
        _write_marker(tmp_path)

        with pytest.raises(ValueError, match="unknown task"):
            pipeline(cast(Any, "from_repository"), model=tmp_path)
