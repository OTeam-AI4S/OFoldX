# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Recursive persistence and placement of named Pipeline components.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import NamedTuple, Self, cast

import torch

from ..hub import ResolvedRepository, load_json_object, resolve_repository
from ..modules.backend import Backend
from .base import Pipeline
from .paired import PairedComponent
from .registry import PIPELINES


class ComponentSpec(NamedTuple):
    r"""One named role, repository directory and declared component type.

    Attributes:
        role: Semantic constructor argument identifying the child.
        subdirectory: Nonempty repository-relative component directory.
        component_type: Declared child class for this role. Pipelines load their
            saved IDs; non-executable pairs use this class for reconstruction.
    """

    role: str
    subdirectory: str
    component_type: type[Pipeline] | type[PairedComponent]


class CompositePipeline(Pipeline):
    r"""A workflow owning named, independently loadable components.

    Concrete constructors receive children by semantic role. Saved children keep
    their own workflow identity, including when a child is itself a composition.
    Construction places Models in paired children, including nested children, in
    evaluation mode. This does not disable gradients; callers may change the mode
    explicitly after construction.
    """

    component_layout: tuple[ComponentSpec, ...]
    _source: ResolvedRepository | None

    def __init__(self, components: Mapping[str, Pipeline | PairedComponent]) -> None:
        self._components = dict(components)
        self._source = None
        pending = list(self._components.values())
        visited: set[int] = set()
        while pending:
            child = pending.pop()
            if id(child) in visited:
                continue
            visited.add(id(child))
            if isinstance(child, PairedComponent):
                child.model.eval()
            elif isinstance(child, CompositePipeline):
                pending.extend(child.components.values())

    @property
    def components(self) -> Mapping[str, Pipeline | PairedComponent]:
        r"""Return children keyed by their semantic role."""

        return self._components

    @property
    def source(self) -> ResolvedRepository | None:
        r"""Return the repository snapshot that supplied this workflow."""

        return self._source

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Move every child to the requested device and floating-point dtype."""

        for component in self._components.values():
            component.to(device=device, dtype=dtype)
        return self

    def set_backend(self, backend: Backend) -> Self:
        r"""Select the operator backend for every child."""

        for component in self._components.values():
            component.set_backend(backend)
        return self

    def save(
        self,
        directory: str | PathLike[str],
        *,
        max_shard_size: int | str = "5GB",
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        r"""Save the workflow and its complete children, then upload if requested."""

        pipeline_id = self.id
        components: dict[str, dict[str, str]] = {}
        for slot in self.component_layout:
            if slot.subdirectory == "":
                raise ValueError("a child component requires its own nonempty directory")
            child = self._components[slot.role]
            values = {"directory": slot.subdirectory}
            if isinstance(child, Pipeline):
                values["pipeline"] = child.id
            components[slot.role] = values

        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "pipeline.json"
        marker.unlink(missing_ok=True)
        for slot in self.component_layout:
            self._components[slot.role].save(destination / slot.subdirectory, max_shard_size=max_shard_size)
        marker.write_text(
            json.dumps({"pipeline": pipeline_id, "components": components}, indent=2) + "\n",
            encoding="utf-8",
        )
        if repo_id is not None:
            from ..hub import upload_repository

            upload_repository(
                destination,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )

    @classmethod
    def _load_component(
        cls,
        resolved: ResolvedRepository,
        slot: ComponentSpec,
        values: Mapping[str, str],
        *,
        token: str | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
        backend: Backend | None,
    ) -> Pipeline | PairedComponent:
        directory = values["directory"]
        source = ResolvedRepository(
            resolved.path / directory,
            resolved.repository,
            resolved.commit,
            "/".join(part for part in (resolved.subdirectory, directory) if part),
        )
        component_class = (
            PIPELINES.lookup_pipeline(values["pipeline"]).pipeline
            if "pipeline" in values or issubclass(slot.component_type, Pipeline)
            else slot.component_type
        )
        return cast(
            Pipeline | PairedComponent,
            component_class.from_pretrained(
                source,
                token=token,
                device=device,
                dtype=dtype,
                backend=backend,
            ),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Self:
        r"""Load every named child from one repository snapshot."""

        resolved = resolve_repository(
            pretrained_model_name_or_path, revision=revision, allow_patterns=("pipeline.json",), token=token
        )
        values = cast(dict[str, dict[str, str]], load_json_object(resolved.path / "pipeline.json")["components"])
        components = {
            slot.role: cls._load_component(
                resolved,
                slot,
                values[slot.role],
                token=token,
                device=device,
                dtype=dtype,
                backend=backend,
            )
            for slot in cls.component_layout
        }
        pipeline = cast(Callable[..., Self], cls)(**components)
        pipeline._source = resolved
        return pipeline

    @classmethod
    def load_component(
        cls,
        repository: str | PathLike[str] | ResolvedRepository,
        *,
        role: str,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Pipeline | PairedComponent:
        r"""Load one named child without loading the remaining workflow."""

        slots = tuple(slot for slot in cls.component_layout if slot.role == role)
        if len(slots) != 1:
            raise ValueError(f"{cls.__name__} has no unique component role {role!r}")
        resolved = resolve_repository(repository, revision=revision, allow_patterns=("pipeline.json",), token=token)
        values = cast(dict[str, dict[str, str]], load_json_object(resolved.path / "pipeline.json")["components"])
        return cls._load_component(
            resolved,
            slots[0],
            values[role],
            token=token,
            device=device,
            dtype=dtype,
            backend=backend,
        )
