r"""Model and Processor registration."""

from __future__ import annotations

from collections.abc import Collection
from importlib import import_module
from typing import Any, Literal, NamedTuple

from torch import nn

_STANDARD_TASKS = frozenset({"fold", "design", "score", "pack"})


class _ModelRegistration(NamedTuple):
    r"""Classes and standard tasks belonging to one model identity."""

    id: str
    model: type[nn.Module]
    processor: type[Any]
    model_config: type[Any]
    standard_tasks: frozenset[Literal["fold", "design", "score", "pack"]]


_MODELS: dict[str, _ModelRegistration] = {}
_MODELS_BY_CONFIG: dict[type[Any], _ModelRegistration] = {}
_MODELS_BY_PAIR: dict[tuple[type[nn.Module], type[Any]], _ModelRegistration] = {}

_BUILTIN_MODEL_MODULES = {
    "ofoldx.alphafold2": "ofoldx.models.alphafold2",
    "ofoldx.alphafold2_multimer": "ofoldx.models.alphafold2",
    "ofoldx.alphafold3": "ofoldx.models.alphafold3",
    "ofoldx.boltz1": "ofoldx.models.boltz1",
    "ofoldx.boltz2": "ofoldx.models.boltz2",
    "ofoldx.boltz2_affinity": "ofoldx.models.boltz2",
    "ofoldx.boltzgen": "ofoldx.models.boltzgen",
    "ofoldx.boltzgen_ifold": "ofoldx.models.boltzgen",
    "ofoldx.esmfold2": "ofoldx.models.esmfold2",
    "ofoldx.esmfold2_experimental": "ofoldx.models.esmfold2",
    "ofoldx.ligandmpnn": "ofoldx.models.ligandmpnn",
    "ofoldx.mpnn_packer": "ofoldx.models.mpnn_packer",
    "ofoldx.nampnn": "ofoldx.models.nampnn",
    "ofoldx.odesign": "ofoldx.models.odesign",
    "ofoldx.proteinmpnn": "ofoldx.models.proteinmpnn",
    "ofoldx.protenix": "ofoldx.models.protenix",
    "ofoldx.protenix_constraint": "ofoldx.models.protenix",
    "ofoldx.pxdesign": "ofoldx.models.pxdesign",
    "ofoldx.rfdiffusion": "ofoldx.models.rfdiffusion",
}


def register_model(
    id: str,
    *,
    model: type[nn.Module],
    processor: type[Any],
    standard_tasks: Collection[Literal["fold", "design", "score", "pack"]] = (),
) -> None:
    r"""Register one Model/Processor pair."""

    if not isinstance(id, str):
        raise TypeError("model architecture id must be a string")
    if not id:
        raise ValueError("model architecture id must not be empty")
    if not isinstance(model, type) or not issubclass(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module class")
    if not isinstance(processor, type):
        raise TypeError("processor must be a class")

    model_config = getattr(model, "config_class", None)
    if not isinstance(model_config, type):
        raise TypeError("model.config_class must be a class")

    standard_task_values = frozenset(standard_tasks)
    unknown = standard_task_values - _STANDARD_TASKS
    if unknown:
        raise ValueError(f"unknown standard tasks: {unknown!r}")

    # Reserve every built-in architecture id for its owner module. Without this a pre-import
    # register_model("ofoldx.<builtin>", Fake) would win first-writer-wins and _registration would
    # return the squatter before ever importing the official owner. A registration is the owner's
    # only when the model class is defined inside the owner package (its __module__ is the owner
    # module or a submodule of it); any other caller is rejected. New third-party ids are unaffected.
    owner_module = _BUILTIN_MODEL_MODULES.get(id)
    if owner_module is not None:
        origin = getattr(model, "__module__", "") or ""
        if origin != owner_module and not origin.startswith(f"{owner_module}."):
            raise ValueError(
                f"model architecture {id!r} is a reserved built-in id owned by {owner_module!r}; "
                "third-party models must register under a new id"
            )

    if id in _MODELS:
        raise ValueError(f"model architecture {id!r} is already registered")
    pair = (model, processor)
    if pair in _MODELS_BY_PAIR:
        raise ValueError("Model/Processor pair is already registered")
    owner = _MODELS_BY_CONFIG.get(model_config)
    if owner is not None and owner.model is not model:
        raise ValueError(f"Model config class is already registered as {owner.id!r}")

    registration = _ModelRegistration(
        id=id,
        model=model,
        processor=processor,
        model_config=model_config,
        standard_tasks=standard_task_values,
    )
    _MODELS[id] = registration
    _MODELS_BY_PAIR[pair] = registration
    _MODELS_BY_CONFIG.setdefault(model_config, registration)


def _registration(id: str) -> _ModelRegistration:
    r"""Resolve one explicitly registered model identity."""

    registered = _MODELS.get(id)
    module = _BUILTIN_MODEL_MODULES.get(id)
    if registered is None and module is not None:
        import_module(module)
        registered = _MODELS.get(id)
    if registered is not None:
        return registered
    raise ValueError(f"unknown model architecture: {id!r}")


def _registration_for_pair(model: object, processor: object) -> _ModelRegistration:
    r"""Return the registration for an exact Model/Processor class pair."""

    try:
        return _MODELS_BY_PAIR[type(model), type(processor)]
    except KeyError:
        raise ValueError("model and processor are not one registered architecture pair") from None


def _registration_for_model_config(config: object) -> _ModelRegistration:
    r"""Return the registration selected by an exact Model config class."""

    try:
        return _MODELS_BY_CONFIG[type(config)]
    except KeyError:
        raise ValueError(f"unregistered Model config class: {type(config).__name__}") from None
