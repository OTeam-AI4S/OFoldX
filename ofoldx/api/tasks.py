r"""
The four standard one-shot tasks and their repository-backed factory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias, cast

import torch

from ..data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import ComponentReference, MolecularSystem, PolymerSequence
from ..data.structure import Structure
from ..hub import load_json_object, resolve_hub_repository, resolve_repository
from ..pipelines.registry import PIPELINES

#: Standard task name understood by the high-level task API.
Task = Literal["fold", "design", "score", "pack"]
#: All standard task names.
TASKS: frozenset[Task] = frozenset({"fold", "design", "score", "pack"})

#: Inputs accepted by the high-level [`fold`][] task.
FoldInput: TypeAlias = str | PathLike[str] | PolymerSequence | MolecularSystem | Structure
#: Inputs accepted by the high-level [`design`][] task.
DesignInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str] | None
#: Inputs accepted by the high-level [`score`][] task.
ScoreInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str]
#: Inputs accepted by the high-level [`pack`][] task.
PackInput: TypeAlias = Structure | str | PathLike[str]
#: MSA evidence accepted by folding and design tasks.
MSAInput: TypeAlias = MSA | MSACollection
#: Template evidence accepted by folding and design tasks.
TemplateInput: TypeAlias = Template


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
    pipeline_id = cast(str, load_json_object(marker)["pipeline"])
    selected_pipeline_class: Any = PIPELINES.lookup_pipeline(pipeline_id)
    if not callable(getattr(selected_pipeline_class, task, None)):
        raise ValueError(f"pipeline {pipeline_id!r} does not support {task}")

    snapshot = metadata_repository.path
    if metadata_repository.repository is not None:
        snapshot = resolve_hub_repository(
            metadata_repository.repository,
            revision=metadata_repository.commit,
            allow_patterns=None,
            token=token,
        ).path
    return selected_pipeline_class.from_repository(
        snapshot,
        token=token,
        device=device,
        dtype=dtype,
    )


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
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Predict a structure with a repository-backed folding pipeline.

    Args:
        input: Sequence, molecular system, structure, or supported input file.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        msa: Optional MSA evidence.
        templates: Optional structural templates.
        num_samples: Number of structures to sample when supported by the model.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.

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
    arguments: dict[str, object] = {"num_samples": num_samples}
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
    conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
    num_candidates: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Generate molecular design candidates with a repository-backed pipeline.

    Args:
        input: Molecular system, structure, or supported input file.
        model: Local model repository or Hugging Face Hub repository ID.
        revision: Hub revision to load.
        token: Hugging Face access token used for private repositories.
        device: Device onto which model parameters are loaded.
        dtype: Floating-point dtype used for inference.
        msa: Optional MSA evidence.
        templates: Optional structural templates.
        conditions: Allowed or tied residue-identity constraints.
        num_candidates: Number of independently generated candidates.
        seed: Seed for stochastic inference.
        generator: PyTorch random generator for stochastic inference.

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
    arguments: dict[str, object] = {}
    if num_candidates != 1:
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

    Returns:
        The scoring result produced by the selected pipeline.
    """

    arguments: dict[str, object] = {}
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

    Returns:
        The side-chain packing result produced by the selected pipeline.
    """

    arguments: dict[str, object] = {
        "sequence": sequence,
        "pack_positions": pack_positions,
        "num_samples": num_samples,
    }
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
