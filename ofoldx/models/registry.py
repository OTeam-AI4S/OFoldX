r"""Model and Processor registration."""

from __future__ import annotations

from collections.abc import Collection
from importlib import import_module
from typing import Any, Literal, NamedTuple

from torch import nn

_STANDARD_TASKS = frozenset({"fold", "design", "score", "pack"})


class ModelRegistration(NamedTuple):
    r"""Classes and standard tasks belonging to one model identity."""

    id: str
    model: type[nn.Module]
    processor: type[Any]
    model_config: type[Any]
    standard_tasks: frozenset[Literal["fold", "design", "score", "pack"]]


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


class ModelRegistry(dict):
    r"""Registry of Model/Processor architecture identities, keyed by architecture id.

    ``MODELS.register(id, model=..., processor=..., standard_tasks=...)`` registers
    one identity. Lookups resolve by id (:meth:`lookup`, lazy-importing the owner of
    a built-in id), by exact Model config class (:meth:`for_config`), or by exact
    ``(Model, Processor)`` class pair (:meth:`for_pair`); the config and pair indexes
    keep those lookups exact rather than a linear scan.
    """

    def __init__(self) -> None:
        super().__init__()
        self.by_config: dict[type[Any], ModelRegistration] = {}
        self.by_pair: dict[tuple[type[nn.Module], type[Any]], ModelRegistration] = {}

    def register(
        self,
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
        # MODELS.register("ofoldx.<builtin>", Fake) would win first-writer-wins and lookup would
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

        if id in self:
            raise ValueError(f"model architecture {id!r} is already registered")
        pair = (model, processor)
        if pair in self.by_pair:
            raise ValueError("Model/Processor pair is already registered")
        config_owner = self.by_config.get(model_config)
        if config_owner is not None and config_owner.model is not model:
            raise ValueError(f"Model config class is already registered as {config_owner.id!r}")

        registration = ModelRegistration(
            id=id,
            model=model,
            processor=processor,
            model_config=model_config,
            standard_tasks=standard_task_values,
        )
        self[id] = registration
        self.by_pair[pair] = registration
        self.by_config.setdefault(model_config, registration)

    def lookup(self, id: str) -> ModelRegistration:
        r"""Resolve one registered model identity, importing a built-in owner on demand."""

        if id not in self and id in _BUILTIN_MODEL_MODULES:
            import_module(_BUILTIN_MODEL_MODULES[id])
        try:
            return self[id]
        except KeyError:
            raise ValueError(f"unknown model architecture: {id!r}") from None

    def for_config(self, config: object) -> ModelRegistration:
        r"""Return the registration selected by an exact Model config class."""

        try:
            return self.by_config[type(config)]
        except KeyError:
            raise ValueError(f"unregistered Model config class: {type(config).__name__}") from None

    def for_pair(self, model: object, processor: object) -> ModelRegistration:
        r"""Return the registration for an exact Model/Processor class pair."""

        try:
            return self.by_pair[type(model), type(processor)]
        except KeyError:
            raise ValueError("model and processor are not one registered architecture pair") from None


MODELS = ModelRegistry()
