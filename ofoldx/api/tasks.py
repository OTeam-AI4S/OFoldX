r"""
The four standard one-shot tasks and their repository-backed factory.
"""

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

#: Inputs accepted by the high-level [`fold`][] task.
FoldInput: TypeAlias = (
    str
    | PathLike[str]
    | PolymerSequence
    | Sequence[PolymerSequence]
    | Mapping[str, PolymerSequence]
    | MolecularSystem
    | Structure
)
#: Inputs accepted by the high-level [`design`][] task.
DesignInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str] | None
#: Inputs accepted by the high-level [`score`][] task.
ScoreInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str]
#: Inputs accepted by the high-level [`pack`][] task.
PackInput: TypeAlias = Structure | str | PathLike[str]
#: MSA evidence accepted by folding and design tasks.
MSAInput: TypeAlias = MSA | MSACollection | str | PathLike[str]
#: Template evidence accepted by folding and design tasks.
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
        The pipeline registered for the repository architecture and task.

    Raises:
        ValueError: If the task is unknown or unsupported by the repository.
        TypeError: If repository metadata is malformed.
    """

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
    r"""
    Predict a structure with a repository-backed folding pipeline.

    Args:
        input: Sequence, sequence collection, molecular system, structure, or supported
            input file.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        msa: Optional MSA value or supported MSA file.
        templates: Optional structural templates or template files.
        num_samples: Number of structures to sample when supported by the model.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.
        **task_kwargs: Additional arguments accepted by the selected folding pipeline.

    Returns:
        The folding result produced by the selected pipeline.
    """

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
    r"""
    Generate molecular design candidates with a repository-backed pipeline.

    Args:
        input: Molecular system, structure, supported input file, or ``None`` for
            pipelines that support unconditional generation.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        msa: Optional MSA value or supported MSA file.
        templates: Optional structural templates or template files.
        conditions: Task-specific design conditions.
        num_candidates: Number of independently generated candidates.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.
        **task_kwargs: Additional arguments accepted by the selected design pipeline.

    Returns:
        The design result produced by the selected pipeline.
    """

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
    r"""
    Score a molecular input with a repository-backed pipeline.

    Args:
        input: Molecular system, structure, or supported structure file.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.
        **task_kwargs: Additional arguments accepted by the selected scoring pipeline.

    Returns:
        The scoring result produced by the selected pipeline.
    """

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
    r"""
    Pack side chains with a repository-backed pipeline.

    Args:
        input: Structure or supported structure file.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        sequence: Optional replacement sequence, globally or by instance.
        pack_positions: Optional component occurrences to repack.
        num_samples: Number of packed structures to sample.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.
        **task_kwargs: Additional arguments accepted by the selected packing pipeline.

    Returns:
        The side-chain packing result produced by the selected pipeline.
    """

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
