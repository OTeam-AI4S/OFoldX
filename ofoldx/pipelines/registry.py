r"""
Concrete Pipeline registration.
"""

from __future__ import annotations

from importlib import import_module
from typing import cast

import chanfig

from ..models.registry import BUILTIN_MODULES


class PipelineRegistry(chanfig.Registry):
    r"""
    Registry of concrete task pipelines.
    """

    ids: dict[type[object], str]

    def __init__(self) -> None:
        super().__init__(case_sensitive=True)
        self.setattr("separator", "\0")
        self.setattr("ids", {})

    def register(
        self,
        id: str,
        *,
        pipeline: type[object],
    ) -> None:
        r"""
        Register one concrete Pipeline class.

        Args:
            id: Stable pipeline identifier.
            pipeline: Concrete Pipeline class.
        """

        if id == "":
            raise ValueError("pipeline id must be a nonempty string")

        self._import_builtin(id)

        if pipeline in self.ids:
            raise ValueError(f"pipeline class is already registered as {self.ids[pipeline]!r}")

        super().register(pipeline, id)
        self.ids[pipeline] = id

    def _import_builtin(self, id: str) -> None:
        module = BUILTIN_MODULES.get(id)
        if module is not None and id not in self:
            import_module(module)

    def lookup_pipeline(self, id: str) -> type[object]:
        r"""
        Return the concrete class registered for a pipeline ID.

        Args:
            id: Registered pipeline ID.

        Returns:
            The matching pipeline class.
        """

        self._import_builtin(id)
        return cast(type[object], super().lookup(id))

    def id_for(self, pipeline: type[object]) -> str | None:
        r"""
        Return a pipeline class's registered ID.

        Args:
            pipeline: Concrete pipeline class.

        Returns:
            Its registered ID, or ``None`` when unregistered.
        """

        return self.ids.get(pipeline)


#: Global registry of OFoldX task pipelines.
PIPELINES = PipelineRegistry()
