r"""Model and Processor registration."""

from __future__ import annotations

from collections.abc import Collection
from importlib import import_module
from typing import Literal, NamedTuple, cast

import chanfig

from ..model import Model
from ..processor import Processor

Task = Literal["fold", "design", "score", "pack"]
TASKS: frozenset[Task] = frozenset({"fold", "design", "score", "pack"})


class ModelRegistration(NamedTuple):
    r"""Classes and standard tasks belonging to one model identity."""

    id: str
    model: type[Model]
    processor: type[Processor]
    processor_config: type[chanfig.Config] | None
    model_config: type[chanfig.Config]
    standard_tasks: frozenset[Task]


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


class ModelRegistry(chanfig.Registry):
    r"""Registry of Model/Processor architecture identities, keyed by architecture id.

    ``MODELS.register_model(id, model=..., processor=..., processor_config=...,
    standard_tasks=...)`` registers one identity. Lookups resolve by id
    (:meth:`lookup_model`, lazy-importing the owner of a built-in id), by exact Model
    config class (:meth:`for_config`), or by exact ``(Model, Processor)`` class pair
    (:meth:`for_pair`); the config and pair indexes keep those lookups exact rather than
    a linear scan.
    """

    by_config: dict[type[chanfig.Config], ModelRegistration]
    by_pair: dict[tuple[type[Model], type[Processor]], ModelRegistration]

    def __init__(self) -> None:
        super().__init__(case_sensitive=True)
        self.setattr("separator", "\0")
        self.setattr("by_config", {})
        self.setattr("by_pair", {})

    def register_model(
        self,
        id: str,
        *,
        model: type[Model],
        processor: type[Processor],
        processor_config: type[chanfig.Config] | None,
        standard_tasks: Collection[str] = (),
    ) -> None:
        r"""Register one Model/Processor pair."""

        self._import_builtin(id)

        model_config = model.config_class

        standard_task_values = frozenset(standard_tasks)
        unknown = standard_task_values - TASKS
        if unknown:
            raise ValueError(f"unknown standard tasks: {unknown!r}")

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
            processor_config=processor_config,
            model_config=model_config,
            standard_tasks=cast(frozenset[Task], standard_task_values),
        )
        super().register(registration, id)
        self.by_pair[pair] = registration
        self.by_config.setdefault(model_config, registration)

    def _import_builtin(self, id: str) -> None:
        module = _BUILTIN_MODEL_MODULES.get(id)
        if module is not None and id not in self:
            import_module(module)

    def lookup_model(self, id: str) -> ModelRegistration:
        r"""Resolve one registered model identity, importing a built-in owner on demand."""

        self._import_builtin(id)
        return cast(ModelRegistration, super().lookup(id))

    def for_config(self, config: chanfig.Config) -> ModelRegistration:
        r"""Return the registration selected by an exact Model config class."""

        return self.by_config[cast(type[chanfig.Config], type(config))]

    def for_pair(self, model: Model, processor: Processor) -> ModelRegistration:
        r"""Return the registration for an exact Model/Processor class pair."""

        return self.by_pair[type(model), type(processor)]


MODELS = ModelRegistry()
