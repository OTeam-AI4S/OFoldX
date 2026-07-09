r"""
Task verbs -- the Tier-0 front door.

``fold`` / ``design`` / ``score`` / ``pack`` are the one public name for each act; each loads the task
pipeline for ``model`` and runs it. ``pipeline(task, model)`` returns the ready task pipeline for
lower-level control. ``inputs`` is a parsed [`Structure`][ofoldx.data.structure.Structure], a structure
file path, or an already-typed model input -- the same inputs the pipelines accept. ``model`` is a local
directory / Hub id, or an already-loaded pipeline to reuse across calls.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any

from .artifacts import normalize_task
from .models.capability import Designer, Folder, Packer, Scorer
from .pipelines import (
    AllAtomDesignPipeline,
    BackboneDesignPipeline,
    BindingAffinityPipeline,
    DesignPipeline,
    Pipeline,
    SequenceDesignPipeline,
    SideChainPackingPipeline,
    StructurePredictionPipeline,
)
from .results import DesignResult, StructurePredictionResult

if TYPE_CHECKING:
    from .models.contract import BindingAffinityOutput

_PIPELINE_FOR_TASK: dict[str, type[Pipeline[Any, Any]]] = {
    "structure_prediction": StructurePredictionPipeline,
    "fold": StructurePredictionPipeline,
    "design": DesignPipeline,
    "sequence_design": SequenceDesignPipeline,
    "backbone_design": BackboneDesignPipeline,
    "all_atom_design": AllAtomDesignPipeline,
    "binding_affinity": BindingAffinityPipeline,
    "score": BindingAffinityPipeline,
    "side_chain_packing": SideChainPackingPipeline,
    "pack": SideChainPackingPipeline,
}


def _pipeline_class(task: str) -> type[Pipeline[Any, Any]]:
    key = normalize_task(task)
    try:
        return _PIPELINE_FOR_TASK[key]
    except KeyError:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(set(_PIPELINE_FOR_TASK))}") from None


def pipeline(task: str, model: str | PathLike[str] | None = None, **from_pretrained_kwargs: Any) -> Pipeline[Any, Any]:
    r"""
    Load the task pipeline for ``model``.

    Resolves ``task`` (a canonical task name or a verb: ``fold``/``design``/``score``/``pack``) to its
    pipeline class and loads ``model`` through it, which also asserts the artifact belongs to that task.

    Args:
        task: Task name or verb.
        model: Local pretrained directory or Hugging Face Hub id.
        **from_pretrained_kwargs: Forwarded to the pipeline's ``from_pretrained``.

    Returns:
        The ready task pipeline.

    Examples:
        >>> import ofoldx
        >>> pipe = ofoldx.pipeline("fold", "oteam/alphafold3")  # doctest: +SKIP
        >>> pipe("complex.cif")  # doctest: +SKIP
    """

    return _pipeline_class(task).from_pretrained(_resolve_model(task, model), **from_pretrained_kwargs)


# Named model catalog: task -> default Hub id, keyed by pipeline class so a task and its verb alias
# ("design" / "fold") share one default. Empty until weights are published; populate with
# register_default_model so ``ofoldx.fold("seq")`` can omit ``model=``.
_DEFAULT_MODEL: dict[type, str] = {}


def register_default_model(task: str, model: str) -> None:
    r"""Register the default model id for a task/verb, enabling ``model=None`` calls."""
    _DEFAULT_MODEL[_pipeline_class(task)] = model


def _resolve_model(task: str, model: Any) -> Any:
    if model is not None:
        return model
    cls = _pipeline_class(task)
    if cls in _DEFAULT_MODEL:
        return _DEFAULT_MODEL[cls]
    raise ValueError(
        f"no model given and no default registered for task {task!r}; "
        f"pass model=... or call ofoldx.register_default_model({task!r}, <id>)"
    )


def _provider(task: str, model: Any, capability: type) -> Any:
    model = _resolve_model(task, model)
    provider = _pipeline_class(task).from_pretrained(model) if isinstance(model, (str, PathLike)) else model
    if not isinstance(provider, capability):
        raise TypeError(
            f"{type(provider).__name__} does not implement the {task!r} capability " f"({capability.__name__}.{task})"
        )
    return provider


def _source_structure(inputs: Any, provider: Any) -> Any:
    r"""
    Return the atom-ordered reference [`Structure`][ofoldx.data.structure.Structure] a result reconstructs onto.

    The provider's processor is asked first for a ``source_structure(inputs)`` hook -- it knows the exact atom
    layout its predicted coordinates use (the materialized heavy-atom set for a structure model, the residue
    atom set for a sequence model). Absent a hook, a Structure/path input is read directly. ``None`` when no
    reference is available, so ``save_pdb`` raises a clear error rather than mis-aligning coordinates.
    """

    from .data.structure import Structure
    from .io import read_structure

    processor = getattr(provider, "processor", None)
    builder = getattr(processor, "source_structure", None)
    if callable(builder):
        try:
            structure = builder(inputs)
        except Exception:  # noqa: BLE001 - a family that cannot build a reference structure just yields None
            structure = None
        if structure is not None:
            return structure
    if isinstance(inputs, Structure):
        return inputs
    if isinstance(inputs, (str, PathLike)):
        try:
            return read_structure(inputs)
        except (OSError, ValueError):
            return None
    return None


def fold(inputs: Any, *, model: Any = None, **kwargs: Any) -> StructurePredictionResult:
    r"""
    Predict a structure for ``inputs`` using ``model``.

    Returns a [`StructurePredictionResult`][ofoldx.results.StructurePredictionResult] exposing
    ``.plddt`` / ``.coords`` / ``.save_pdb`` with the raw typed output on ``.raw``.

    Examples:
        >>> import ofoldx
        >>> ofoldx.fold("complex.cif", model="oteam/alphafold3").save_pdb("out.pdb")  # doctest: +SKIP
    """

    provider = _provider("fold", model, Folder)
    output = provider.fold(inputs, **kwargs)
    return StructurePredictionResult(output, source=_source_structure(inputs, provider))


def design(inputs: Any, *, model: Any = None, **kwargs: Any) -> DesignResult:
    r"""
    Design sequence(s) or backbone(s) for ``inputs`` using ``model``.

    Returns a [`DesignResult`][ofoldx.results.DesignResult] wrapping the ranked candidates.

    Examples:
        >>> import ofoldx
        >>> ofoldx.design("target.cif", model="oteam/proteinmpnn")  # doctest: +SKIP
    """

    provider = _provider("design", model, Designer)
    output = provider.design(inputs, **kwargs)
    return DesignResult(output, source=_source_structure(inputs, provider))


def score(inputs: Any, *, model: Any = None, **kwargs: Any) -> BindingAffinityOutput:
    r"""
    Score ``inputs`` (e.g. binding affinity) using ``model``.

    Examples:
        >>> import ofoldx
        >>> ofoldx.score("complex.cif", model="oteam/boltz2-affinity")  # doctest: +SKIP
    """

    return _provider("score", model, Scorer).score(inputs, **kwargs)


def pack(inputs: Any, *, model: Any = None, **kwargs: Any) -> dict[str, Any]:
    r"""
    Pack side chains for ``inputs`` using ``model``.

    Examples:
        >>> import ofoldx
        >>> ofoldx.pack("backbone.cif", model="oteam/ligandmpnn-packer")  # doctest: +SKIP
    """

    return _provider("pack", model, Packer).pack(inputs, **kwargs)
