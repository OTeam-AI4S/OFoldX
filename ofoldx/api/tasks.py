r"""
Repository-backed Pipeline selection.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Literal

import torch

from ..hub import load_json_object, resolve_repository
from ..pipelines.registry import PIPELINES

#: Standard task name understood by the high-level task API.
Task = Literal["fold", "design", "score", "pack"]
#: All standard task names.
TASKS: frozenset[Task] = frozenset({"fold", "design", "score", "pack"})


def pipeline(
    task: Task,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    r"""
    Load a task pipeline from a model repository.

    Args:
        task: Standard task name: ``"fold"``, ``"design"``, ``"score"``, or
            ``"pack"``.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype applied to model parameters and processor outputs.

    Returns:
        The Pipeline named by the repository marker.

    Raises:
        FileNotFoundError: If the repository has no ``pipeline.json`` marker.
        ValueError: If the task is unknown or unsupported by the selected Pipeline.
        TypeError: If repository metadata is malformed.
    """

    if task not in TASKS:
        valid = ", ".join(repr(name) for name in sorted(TASKS))
        raise ValueError(f"unknown task {task!r}; valid tasks are {valid}") from None
    metadata_repository = resolve_repository(
        model,
        revision=revision,
        token=token,
        allow_patterns=("pipeline.json",),
    )
    marker = metadata_repository.path / "pipeline.json"
    pipeline_id = load_json_object(marker).get("pipeline")
    if not isinstance(pipeline_id, str):
        raise TypeError("pipeline.json pipeline must be a string")
    selected_pipeline_class: Any = PIPELINES.lookup_pipeline(pipeline_id)
    if not callable(getattr(selected_pipeline_class, task, None)):
        raise ValueError(f"pipeline {pipeline_id!r} does not support {task}")

    selected_repository: str | PathLike[str] = metadata_repository.path
    selected_revision: str | None = None
    if metadata_repository.repository is not None:
        selected_repository = metadata_repository.repository
        selected_revision = metadata_repository.commit
    return selected_pipeline_class.from_repository(
        selected_repository,
        revision=selected_revision,
        token=token,
        device=device,
        dtype=dtype,
    )
