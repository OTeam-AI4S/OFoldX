r"""The four standard one-shot tasks and their repository-backed factory."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias

import chanfig
import torch

from ..data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..data.evidence import MSA, MSACollection, Template
from ..data.molecular import ComponentReference, MolecularSystem, PolymerSequence
from ..data.structure import Structure
from ..hub import (
    _PROCESSOR_METADATA,
    _hydrate_weights,
    _load_registered_model,
    _load_registered_processor,
    _model_config,
    _resolve_hub_repository,
    _resolve_repository,
)
from ..pipelines import DesignPipeline, FoldPipeline, PackPipeline, ScorePipeline
from ..pipelines.registry import PIPELINES

FoldInput: TypeAlias = str | PathLike[str] | PolymerSequence | MolecularSystem | Structure
DesignInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str]
ScoreInput: TypeAlias = MolecularSystem | Structure | str | PathLike[str]
PackInput: TypeAlias = Structure | str | PathLike[str]
MSAInput: TypeAlias = MSA | MSACollection
TemplateInput: TypeAlias = Template

_STANDARD_PIPELINE_CLASSES = {
    "fold": FoldPipeline,
    "design": DesignPipeline,
    "score": ScorePipeline,
    "pack": PackPipeline,
}


def _validate_input(input: object) -> None:
    r"""Reject a malformed one-shot input at the API boundary with a clear error.

    The one-shot task functions accept an in-memory carrier (:class:`Structure`,
    :class:`MolecularSystem`, :class:`PolymerSequence`) or a path to an existing structure file.
    Anything else — ``None``, an empty string, a directory, an arbitrary object — is caught here
    so the caller gets a clear ``TypeError`` / ``ValueError`` instead of an opaque failure deep
    inside a Processor. A path that simply does not exist is left to the Processor's own
    ``FileNotFoundError``.
    """

    if isinstance(input, (Structure, MolecularSystem, PolymerSequence)):
        return
    if isinstance(input, (str, PathLike)):
        text = os.fspath(input)
        if not text:
            raise ValueError("input path is empty; pass a path to a structure file")
        if Path(text).expanduser().is_dir():
            raise ValueError(f"input path {text!r} is a directory, not a structure file")
        return
    raise TypeError(
        "input must be a Structure, MolecularSystem, PolymerSequence, or a path to a structure "
        f"file, not {type(input).__name__}"
    )


def _accepts_keyword(method: object, name: str) -> bool:
    r"""Report whether ``method`` accepts the keyword ``name`` (directly or via ``**kwargs``)."""

    parameters = inspect.signature(method).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return any(parameter.name == name for parameter in parameters)


def _guard_evidence(method: object, model: object, *, msa: object, templates: Sequence[object]) -> None:
    r"""Fail clearly when evidence is supplied to a model whose task cannot consume it.

    ``fold`` / ``design`` advertise ``msa=`` / ``templates=``, but a structure-only concrete
    pipeline (ProteinMPNN, ...) does not accept them. Rather than forwarding into a raw
    ``TypeError: ... unexpected keyword argument 'msa'`` deep in the call, name the model and the
    evidence its task ignores.
    """

    unsupported: list[str] = []
    if msa is not None and not _accepts_keyword(method, "msa"):
        unsupported.append("MSA")
    if templates and not _accepts_keyword(method, "templates"):
        unsupported.append("templates")
    if unsupported:
        raise TypeError(f"model {model!r} does not consume {'/'.join(unsupported)}")


def pipeline(
    task: Literal["fold", "design", "score", "pack"],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    r"""Load one paired Model and Processor into an explicit task Pipeline."""

    try:
        standard_pipeline_class = _STANDARD_PIPELINE_CLASSES[task]
    except KeyError:
        valid = ", ".join(repr(name) for name in _STANDARD_PIPELINE_CLASSES)
        raise ValueError(f"unknown task {task!r}; valid tasks are {valid}") from None
    metadata_repository = _resolve_repository(
        model,
        revision=revision,
        allow_patterns=(*_PROCESSOR_METADATA, "pipeline.json"),
    )
    marker = metadata_repository.path / "pipeline.json"
    has_marker = marker.exists()
    if has_marker:
        marker_values = chanfig.NestedDict.load(marker)
        if not isinstance(marker_values, dict):
            raise TypeError("pipeline.json must contain a JSON object")
        if set(marker_values) != {"pipeline"}:
            raise ValueError("pipeline.json must contain exactly one pipeline field")
        pipeline_id = marker_values["pipeline"]
        if not isinstance(pipeline_id, str):
            raise TypeError("pipeline.json pipeline must be a string")
        selected_pipeline_class = PIPELINES.lookup(pipeline_id)
        if not callable(getattr(selected_pipeline_class, task, None)):
            raise ValueError(f"pipeline {pipeline_id!r} does not support {task}")
        if getattr(selected_pipeline_class, "_PAIR_LAYOUT", None) is not None:
            # DESIGN §7.4 multi-pair: the concrete class owns several class-owned pairs and must load
            # every one of them from the same immutable snapshot, so the factory dispatches to the
            # class's own multi-pair loader instead of the one-pair (model, processor) call. Pin the
            # full snapshot at the commit the marker was read from; for a local directory that same
            # directory already holds every pair.
            if metadata_repository.repository is None:
                snapshot = metadata_repository.path
            else:
                snapshot = _resolve_hub_repository(
                    metadata_repository.repository,
                    revision=metadata_repository.commit,
                    allow_patterns=None,
                ).path
            return selected_pipeline_class.from_repository(snapshot, device=device, dtype=dtype)
    else:
        selected_pipeline_class = standard_pipeline_class

    registration, model_config = _model_config(metadata_repository.path)
    if not has_marker and task not in registration.standard_tasks:
        raise ValueError(f"model architecture {registration.id!r} does not support standard {task}")
    # Construct the Processor before requesting weights.
    loaded_processor = _load_registered_processor(
        metadata_repository,
        registration,
        device=device,
        dtype=dtype,
    )
    repository = _hydrate_weights(metadata_repository)
    loaded_model = _load_registered_model(
        repository,
        registration,
        model_config,
        device=device,
        dtype=dtype,
    )

    return selected_pipeline_class(model=loaded_model, processor=loaded_processor)


def fold(
    input: FoldInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    msa: MSAInput | None = None,
    templates: Sequence[TemplateInput] = (),
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""Fold a biological input with one repository-backed Model/Processor pair."""

    _validate_input(input)
    fold_pipeline = pipeline("fold", model=model, revision=revision, device=device, dtype=dtype)
    _guard_evidence(fold_pipeline.fold, model, msa=msa, templates=templates)
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
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    msa: MSAInput | None = None,
    templates: Sequence[TemplateInput] = (),
    conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
    num_candidates: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""Design candidates with one repository-backed Model/Processor pair."""

    _validate_input(input)
    design_pipeline = pipeline("design", model=model, revision=revision, device=device, dtype=dtype)
    _guard_evidence(design_pipeline.design, model, msa=msa, templates=templates)
    arguments: dict[str, object] = {"num_candidates": num_candidates}
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
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""Score a molecular input with one repository-backed Model/Processor pair."""

    _validate_input(input)
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
    ).score(input, **arguments)


def pack(
    input: PackInput,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    sequence: str | Mapping[str, str | Sequence[str]] | None = None,
    positions: Sequence[ComponentReference] | None = None,
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""Pack side chains with one repository-backed Model/Processor pair."""

    _validate_input(input)
    arguments: dict[str, object] = {
        "sequence": sequence,
        "positions": positions,
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
    ).pack(input, **arguments)
