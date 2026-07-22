r"""Concrete Pipeline registration."""

from __future__ import annotations

from collections.abc import Collection
from importlib import import_module
from os import PathLike
from typing import NamedTuple, Protocol, cast

import chanfig
import torch

from ..models.registry import TASKS, Task

_BUILTIN_PIPELINE_MODULES = {
    "ofoldx.alphafold2": "ofoldx.models.alphafold2",
    "ofoldx.alphafold2_multimer": "ofoldx.models.alphafold2",
    "ofoldx.alphafold3": "ofoldx.models.alphafold3",
    "ofoldx.boltz1": "ofoldx.models.boltz1",
    "ofoldx.boltz2": "ofoldx.models.boltz2",
    "ofoldx.esmfold2": "ofoldx.models.esmfold2",
    "ofoldx.esmfold2_design": "ofoldx.models.esmfold2",
    "ofoldx.ligandmpnn": "ofoldx.models.ligandmpnn",
    "ofoldx.mpnn_packer": "ofoldx.models.mpnn_packer",
    "ofoldx.nampnn": "ofoldx.models.nampnn",
    "ofoldx.protenix": "ofoldx.models.protenix",
    "ofoldx.protenix_constraint": "ofoldx.models.protenix",
    "ofoldx.proteinmpnn": "ofoldx.models.proteinmpnn",
    "ofoldx.pxdesign": "ofoldx.models.pxdesign",
    "ofoldx.rfdiffusion": "ofoldx.models.rfdiffusion",
    # Multi-pair pipelines register their repository loader.
    "ofoldx.boltz2_affinity": "ofoldx.models.boltz2",
    "ofoldx.odesign": "ofoldx.models.odesign",
    "ofoldx.boltzgen": "ofoldx.models.boltzgen",
}


class RepositoryLoader(Protocol):
    r"""Load a pipeline with a class-defined repository layout."""

    def __call__(
        self,
        repository: str | PathLike[str],
        /,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> object: ...


class PipelineRegistration(NamedTuple):
    r"""One concrete pipeline registration."""

    id: str
    pipeline: type[object]
    tasks: frozenset[Task]
    architectures: frozenset[str]
    repository_loader: RepositoryLoader | None


class PipelineRegistry(chanfig.Registry):
    r"""Registry of concrete pipelines keyed by pipeline ID.

    Each registration binds a class, tasks, architectures, and an optional repository
    loader.
    """

    ids: dict[type[object], str]
    by_architecture: dict[str, PipelineRegistration]

    def __init__(self) -> None:
        super().__init__(case_sensitive=True)
        self.setattr("separator", "\0")
        self.setattr("ids", {})
        self.setattr("by_architecture", {})

    def register_pipeline(
        self,
        id: str,
        *,
        pipeline: type[object],
        tasks: Collection[Task],
        architectures: Collection[str] = (),
        repository_loader: RepositoryLoader | None = None,
    ) -> None:
        r"""Register a concrete pipeline and its architecture routes."""

        self._import_builtin(id)

        if pipeline in self.ids:
            raise ValueError(f"pipeline class is already registered as {self.ids[pipeline]!r}")

        task_values = frozenset(tasks)
        unknown = task_values - TASKS
        if unknown:
            raise ValueError(f"unknown pipeline tasks: {unknown!r}")

        architecture_values = frozenset(architectures)
        for architecture in architecture_values:
            owner = self.by_architecture.get(architecture)
            if owner is not None:
                raise ValueError(f"model architecture {architecture!r} already routes to {owner.id!r}")

        registration = PipelineRegistration(
            id=id,
            pipeline=pipeline,
            tasks=task_values,
            architectures=architecture_values,
            repository_loader=repository_loader,
        )
        super().register(registration, id)
        self.ids[pipeline] = id
        for architecture in architecture_values:
            self.by_architecture[architecture] = registration

    def _import_builtin(self, id: str) -> None:
        module = _BUILTIN_PIPELINE_MODULES.get(id)
        if module is not None and id not in self:
            import_module(module)

    def lookup_registration(self, id: str) -> PipelineRegistration:
        r"""Resolve a registration, importing its built-in module on demand."""

        self._import_builtin(id)
        return cast(PipelineRegistration, super().lookup(id))

    def lookup_pipeline(self, id: str) -> type[object]:
        r"""Resolve one registered Pipeline class, importing a built-in owner on demand."""

        return self.lookup_registration(id).pipeline

    def id_for(self, pipeline: type[object]) -> str | None:
        r"""Return a registered pipeline class's ID, or ``None``."""

        return self.ids.get(pipeline)

    def for_architecture(self, architecture: str) -> PipelineRegistration | None:
        r"""Return the registration for a markerless model architecture."""

        self._import_builtin(architecture)
        return self.by_architecture.get(architecture)


PIPELINES = PipelineRegistry()
