r"""
Concrete Pipeline registration.
"""

from __future__ import annotations

from ..models.registry import import_builtin


class PipelineRegistry:
    r"""
    Registry of exact Pipeline IDs and classes.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, type[object]] = {}
        self._ids: dict[type[object], str] = {}

    def register(self, id: str, *, pipeline: type[object]) -> None:
        r"""
        Register one concrete Pipeline class.

        Args:
            id: Stable Pipeline identifier.
            pipeline: Concrete Pipeline class.
        """

        if id == "":
            raise ValueError("pipeline id must be nonempty")

        self._import_builtin(id)
        if id in self._pipelines:
            raise ValueError(f"pipeline id is already registered: {id!r}")
        if pipeline in self._ids:
            raise ValueError(f"pipeline class is already registered as {self._ids[pipeline]!r}")

        self._pipelines[id] = pipeline
        self._ids[pipeline] = id

    def _import_builtin(self, id: str) -> None:
        if id not in self._pipelines:
            import_builtin(id)

    def lookup_pipeline(self, id: str) -> type[object]:
        r"""
        Return the Pipeline class registered for an exact ID.
        """

        self._import_builtin(id)
        return self._pipelines[id]

    def id_for(self, pipeline: type[object]) -> str:
        r"""
        Return the exact ID registered for a Pipeline class.
        """

        return self._ids[pipeline]


#: Global registry of OFoldX task Pipelines.
PIPELINES = PipelineRegistry()
