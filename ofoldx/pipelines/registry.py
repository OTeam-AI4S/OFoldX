r"""
Concrete Pipeline registration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple

from ..models.registry import import_builtin

PipelineTask = Literal["fold", "design", "score", "pack"]
_PipelineExecutionStage = Literal["prepare", "forward", "decode", "execute"]
_PipelineExecutionOwner = Literal["pipeline", "model"]


class PipelineExecutionStep(NamedTuple):
    r"""
    One explicit callable in a service-side Pipeline execution.

    ``execute`` denotes an indivisible public Pipeline operation. Staged paths
    use ``prepare``, ``forward``, and ``decode`` without method-name lookup.
    """

    stage: _PipelineExecutionStage
    owner: _PipelineExecutionOwner
    function: Callable[..., object]


PipelineExecution = tuple[PipelineExecutionStep, ...]


class PipelineRegistration(NamedTuple):
    r"""One concrete Pipeline and its explicit service execution paths."""

    id: str
    pipeline: type[object]
    executions: Mapping[PipelineTask, PipelineExecution]

    @property
    def tasks(self) -> frozenset[PipelineTask]:
        r"""Return tasks with an explicit execution path."""

        return frozenset(self.executions)


class PipelineRegistry:
    r"""
    Registry of exact Pipeline IDs and classes.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, PipelineRegistration] = {}
        self._pipelines: dict[type[object], PipelineRegistration] = {}

    def register(
        self,
        id: str,
        *,
        pipeline: type[object],
        executions: Mapping[PipelineTask, PipelineExecution],
    ) -> None:
        r"""
        Register one concrete Pipeline class.

        Args:
            id: Stable Pipeline identifier.
            pipeline: Concrete Pipeline class.
            executions: Public tasks and their ordered service-side callables.
        """

        if id == "":
            raise ValueError("pipeline id must be nonempty")

        self._import_builtin(id)
        if id in self._registrations:
            raise ValueError(f"pipeline id is already registered: {id!r}")
        if pipeline in self._pipelines:
            raise ValueError(f"pipeline class is already registered as {self._pipelines[pipeline].id!r}")

        if not executions:
            raise ValueError("pipeline executions must not be empty")
        if any(not execution for execution in executions.values()):
            raise ValueError("pipeline executions must contain at least one step")
        registration = PipelineRegistration(id, pipeline, dict(executions))
        self._registrations[id] = registration
        self._pipelines[pipeline] = registration

    def _import_builtin(self, id: str) -> None:
        if id not in self._registrations:
            import_builtin(id)

    def lookup_pipeline(self, id: str) -> PipelineRegistration:
        r"""
        Return the Pipeline registration for an exact ID.
        """

        self._import_builtin(id)
        return self._registrations[id]

    def id_for(self, pipeline: type[object]) -> str:
        r"""
        Return the exact ID registered for a Pipeline class.
        """

        return self._pipelines[pipeline].id


#: Global registry of OFoldX task Pipelines.
PIPELINES = PipelineRegistry()
