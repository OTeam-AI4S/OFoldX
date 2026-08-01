r"""
Repository-backed Pipeline selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Literal

import torch

from ..data import ComponentReference, MolecularSystem, Structure
from ..hub import load_json_object, resolve_repository
from ..modules.backend import Backend
from ..pipelines.registry import PIPELINES


def pipeline(
    task: Literal["fold", "design", "score", "pack"],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
) -> Any:
    r"""
    Load the requested task Pipeline from a model repository.
    """

    if task not in ("fold", "design", "score", "pack"):
        raise ValueError(f"unknown task {task!r}")

    metadata_repository = resolve_repository(
        model,
        revision=revision,
        token=token,
        allow_patterns=("pipeline.json",),
    )
    pipeline_id = load_json_object(metadata_repository.path / "pipeline.json")["pipeline"]
    if not isinstance(pipeline_id, str):
        raise TypeError("pipeline.json pipeline must be a string")

    pipeline_class: Any = PIPELINES.lookup_pipeline(pipeline_id)
    if not callable(getattr(pipeline_class, task, None)):
        raise ValueError(f"pipeline {pipeline_id!r} does not support {task}")

    selected_repository: str | PathLike[str] = metadata_repository.path
    selected_revision: str | None = None
    if metadata_repository.repository is not None:
        selected_repository = metadata_repository.repository
        selected_revision = metadata_repository.commit
    return pipeline_class.from_repository(
        selected_repository,
        revision=selected_revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )


def score(
    input: MolecularSystem | Structure | str | PathLike[str],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Load a scoring Pipeline and run its scientifically neutral common call.
    """

    selected = pipeline(
        "score",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )
    return selected.score(input, seed=seed, generator=generator)


def pack(
    input: Structure | str | PathLike[str],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    pack_positions: Sequence[ComponentReference] | None = None,
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Load a side-chain packing Pipeline and run its common call.
    """

    selected = pipeline(
        "pack",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )
    return selected.pack(
        input,
        sequence=sequence,
        pack_positions=pack_positions,
        num_samples=num_samples,
        seed=seed,
        generator=generator,
    )
