r"""Runner, objective, and collate registries.

Mirrors the model-side registry pattern: names resolve to objects, and the
config's ``runner`` / ``objective`` keys drive dispatch. ``RUNNERS`` selects the
Runner class from ``config.runner``; ``OBJECTIVES`` and ``COLLATES`` map an
objective key to its loss reducer and (optionally) a family-specific collate.

Objectives are keyed by an **objective name** (``"mpnn"``, ``"diffusion"``), and
:data:`ARCHITECTURE_OBJECTIVES` maps each supported architecture id to its default
objective name. A family with no faithful in-repo training objective is simply
absent from that table, so :func:`build_objective` raises a clear error rather
than fabricating a loss (DESIGN Principle 13).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chanfig

from .config import RunnerConfig
from .objectives.base import Objective


class _RunnerRegistry(chanfig.ConfigRegistry):
    r"""Resolve the Runner class from ``config.runner``."""

    key = "runner"


RUNNERS = _RunnerRegistry()
OBJECTIVES: chanfig.Registry = chanfig.Registry()
COLLATES: chanfig.Registry = chanfig.Registry()

# Architecture id -> default objective name. Only architectures with a grounded,
# in-repo objective appear here; others have no training objective yet.
ARCHITECTURE_OBJECTIVES: dict[str, str] = {
    "ofoldx.proteinmpnn": "mpnn",
    "ofoldx.ligandmpnn": "mpnn",
    "ofoldx.nampnn": "mpnn",
    "ofoldx.boltz1": "diffusion",
    "ofoldx.boltz2": "diffusion",
    "ofoldx.boltzgen": "diffusion",
    "ofoldx.protenix": "diffusion",
}


def register_objective(name: str) -> Callable[[type], type]:
    r"""Register an :class:`Objective` implementation under ``name``."""

    def decorator(objective: type) -> type:
        OBJECTIVES.register(objective, name)
        return objective

    return decorator


def register_collate(name: str) -> Callable[[Callable], Callable]:
    r"""Register a family-specific collate function under an objective ``name``."""

    def decorator(collate: Callable) -> Callable:
        COLLATES.register(collate, name)
        return collate

    return decorator


def _objective_name(architecture: str, config: RunnerConfig) -> str:
    name = config.objective or ARCHITECTURE_OBJECTIVES.get(architecture)
    if name is None:
        raise ValueError(
            f"no training objective is defined for architecture {architecture!r}; "
            f"set `objective` explicitly or use an architecture with a registered objective "
            f"({', '.join(sorted(set(ARCHITECTURE_OBJECTIVES.values())))})"
        )
    return name


def build_objective(architecture: str, config: RunnerConfig) -> Objective:
    r"""Resolve the :class:`Objective` for one architecture + config."""

    from .objectives import diffusion, mpnn  # noqa: F401 — trigger objective registration

    name = _objective_name(architecture, config)
    return OBJECTIVES.build(type=name)


def build_collate(architecture: str, config: RunnerConfig) -> Callable[[list[Any]], Any]:
    r"""Resolve the collate for one architecture + config.

    Returns a family-specific collate when one is registered for the resolved
    objective name, otherwise the dataclass-generic :func:`ofoldx.runners.collate.collate`.
    """

    from .collate import collate as generic_collate
    from .objectives import diffusion, mpnn  # noqa: F401 — trigger collate registration

    name = _objective_name(architecture, config)
    if name in COLLATES:
        return COLLATES.build(type=name)
    return generic_collate
