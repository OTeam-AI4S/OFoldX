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
Common interface for executable biological Pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Callable, ClassVar, Self

from ..hub import ResolvedRepository
from .registry import PIPELINES, PipelineTask


class _TaskCall:
    """Bind a complete call to the current task method, including subclass overrides."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        return getattr(owner if instance is None else instance, self.name)


def task_call(method: Callable[..., Any]) -> _TaskCall:
    return _TaskCall(method.__name__)


class Pipeline(ABC):
    r"""
    An executable workflow with one task and concrete scientific interfaces.

    Concrete Pipelines publish typed ``prepare``, ``forward``, ``postprocess`` and
    complete-call signatures. Prepared inputs retain the correspondence needed by
    postprocessing. Execution may involve sampling, optimization or child Pipelines.

    Attributes:
        task: Biological operation declared by the concrete workflow class.
    """

    task: ClassVar[PipelineTask]

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        r"""Prepare the concrete scientific input and its required correspondence."""

        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Execute the concrete scientific workflow on prepared input."""

        raise NotImplementedError

    @abstractmethod
    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        r"""Produce scientific results using prepared input and execution output."""

        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the complete preparation, execution and postprocessing operation."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self:
        r"""Load this workflow from a native repository."""

        raise NotImplementedError

    @abstractmethod
    def save(self, *args: Any, **kwargs: Any) -> None:
        r"""Save this workflow and its scientific components."""

        raise NotImplementedError

    @abstractmethod
    def to(self, *args: Any, **kwargs: Any) -> Self:
        r"""Place this workflow's scientific components on the requested device and dtype."""

        raise NotImplementedError

    @abstractmethod
    def set_backend(self, *args: Any, **kwargs: Any) -> Self:
        r"""Select the operator backend used by this workflow."""

        raise NotImplementedError

    @property
    def id(self) -> str:
        r"""Return the registered identifier of this concrete workflow."""

        return PIPELINES.id_for(type(self))

    @property
    def components(self) -> Mapping[str, object]:
        r"""Return named child components; a paired leaf has no children."""

        return {}

    def repository_components(self) -> Iterable[tuple[str, object]]:
        r"""Iterate children while collecting repository provenance."""

        return self.components.items()

    @property
    def source(self) -> ResolvedRepository | None:
        r"""Return the repository snapshot, or ``None`` for an in-memory workflow."""

        return None
