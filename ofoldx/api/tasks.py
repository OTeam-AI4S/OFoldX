r"""The four standard one-shot tasks and their repository-backed factory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, TypeAlias

import torch

from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import ComponentReference, MolecularSystem, PolymerSequence
from ..data.structure import Structure
from ..hub import (
    PROCESSOR_METADATA,
    hydrate_weights,
    load_json_object,
    load_registered_model,
    load_registered_processor,
    model_values,
    resolve_hub_repository,
    resolve_repository,
)
from ..models.registry import Task
from ..pipelines import DesignPipeline, FoldPipeline, PackPipeline, ScorePipeline
from ..pipelines.registry import PIPELINES, PipelineRegistration

FoldInput: TypeAlias = (
    str
    | PathLike[str]
    | PolymerSequence
    | Sequence[PolymerSequence]
    | Mapping[str, PolymerSequence]
    | MolecularSystem
    | Structure
)
DesignInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str] | None
ScoreInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str]
PackInput: TypeAlias = Structure | str | PathLike[str]
MSAInput: TypeAlias = MSA | MSACollection | str | PathLike[str]
TemplateInput: TypeAlias = Template | str | PathLike[str]

_STANDARD_PIPELINE_CLASSES = {
    "fold": FoldPipeline,
    "design": DesignPipeline,
    "score": ScorePipeline,
    "pack": PackPipeline,
}


def pipeline(
    task: Task,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    r"""Load one paired Model and Processor into an explicit task Pipeline."""

    try:
        standard_pipeline_class = _STANDARD_PIPELINE_CLASSES[task]
    except KeyError:
        valid = ", ".join(repr(name) for name in _STANDARD_PIPELINE_CLASSES)
        raise ValueError(f"unknown task {task!r}; valid tasks are {valid}") from None
    metadata_repository = resolve_repository(
        model,
        revision=revision,
        token=token,
        allow_patterns=(*PROCESSOR_METADATA, "pipeline.json"),
    )
    marker = metadata_repository.path / "pipeline.json"
    publication: PipelineRegistration | None = None
    if marker.exists():
        marker_values = load_json_object(marker)
        pipeline_id = marker_values.get("pipeline")
        if not isinstance(pipeline_id, str):
            raise TypeError("pipeline.json pipeline must be a string")
        publication = PIPELINES.lookup_registration(pipeline_id)

    registration, raw_model_values = model_values(metadata_repository.path)
    if publication is None:
        publication = PIPELINES.for_architecture(registration.id)
    if publication is None:
        if task not in registration.standard_tasks:
            raise ValueError(f"model architecture {registration.id!r} does not support {task}")
        selected_pipeline_class = standard_pipeline_class
        repository_loader = None
    else:
        if task not in publication.tasks:
            raise ValueError(f"pipeline {publication.id!r} does not support {task}")
        if publication.repository_loader is None and registration.id not in publication.architectures:
            raise ValueError(
                f"pipeline {publication.id!r} is not registered for model architecture {registration.id!r}"
            )
        selected_pipeline_class = publication.pipeline
        repository_loader = publication.repository_loader

    if repository_loader is not None:
        # Composite and externally persisted Pipelines own their complete repository
        # layout. Resolve one immutable snapshot and let the concrete class load it.
        if metadata_repository.repository is None:
            snapshot = metadata_repository.path
        else:
            snapshot = resolve_hub_repository(
                metadata_repository.repository,
                revision=metadata_repository.commit,
                allow_patterns=None,
                token=token,
            ).path
        return repository_loader(snapshot, device=device, dtype=dtype)

    # Construct the Processor before requesting weights.
    loaded_processor = load_registered_processor(
        metadata_repository,
        registration,
        device=device,
        dtype=dtype,
    )
    repository = hydrate_weights(metadata_repository, token=token)
    model_config = registration.model_config(raw_model_values)
    loaded_model = load_registered_model(
        repository,
        registration,
        model_config,
        device=device,
        dtype=dtype,
    )
    loaded_processor.to(device=loaded_model.device, dtype=loaded_model.dtype)

    return selected_pipeline_class(model=loaded_model, processor=loaded_processor)


def fold(
    input: FoldInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    msa: MSAInput | None = None,
    templates: Sequence[TemplateInput] = (),
    num_samples: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    **task_kwargs: Any,
) -> Any:
    r"""Fold a biological input, forwarding task controls to its concrete Pipeline."""

    fold_pipeline = pipeline(
        "fold",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
    )
    arguments: dict[str, object] = dict(task_kwargs)
    if num_samples is not None:
        arguments["num_samples"] = num_samples
    if msa is not None:
        arguments["msa"] = msa
    if templates:
        arguments["templates"] = templates
    if seed is not None:
        arguments["seed"] = seed
    if generator is not None:
        arguments["generator"] = generator
    return fold_pipeline.fold(input, **arguments)


def design(
    input: DesignInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    msa: MSAInput | None = None,
    templates: Sequence[TemplateInput] = (),
    conditions: Sequence[object] = (),
    num_candidates: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    **task_kwargs: Any,
) -> Any:
    r"""Design candidates, forwarding task controls to its concrete Pipeline."""

    design_pipeline = pipeline(
        "design",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
    )
    arguments: dict[str, object] = dict(task_kwargs)
    if num_candidates is not None:
        arguments["num_candidates"] = num_candidates
    if msa is not None:
        arguments["msa"] = msa
    if templates:
        arguments["templates"] = templates
    if conditions:
        arguments["conditions"] = conditions
    if seed is not None:
        arguments["seed"] = seed
    if generator is not None:
        arguments["generator"] = generator
    return design_pipeline.design(input, **arguments)


def score(
    input: ScoreInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    **task_kwargs: Any,
) -> Any:
    r"""Score a molecular input, forwarding task controls to its concrete Pipeline."""

    arguments: dict[str, object] = dict(task_kwargs)
    if seed is not None:
        arguments["seed"] = seed
    if generator is not None:
        arguments["generator"] = generator
    return pipeline(
        "score",
        model=model,
        revision=revision,
        device=device,
        dtype=dtype,
        token=token,
    ).score(input, **arguments)


def pack(
    input: PackInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    pack_positions: Sequence[ComponentReference] | None = None,
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    **task_kwargs: Any,
) -> Any:
    r"""Pack side chains, forwarding task controls to its concrete Pipeline."""

    arguments: dict[str, object] = dict(task_kwargs)
    arguments.update(
        {
            "sequence": sequence,
            "pack_positions": pack_positions,
            "num_samples": num_samples,
        }
    )
    if seed is not None:
        arguments["seed"] = seed
    if generator is not None:
        arguments["generator"] = generator
    return pipeline(
        "pack",
        model=model,
        revision=revision,
        device=device,
        dtype=dtype,
        token=token,
    ).pack(input, **arguments)
