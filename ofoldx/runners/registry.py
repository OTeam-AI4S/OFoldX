r"""Runner and objective registries.

``RUNNERS`` resolves the Runner class from ``config.runner`` — the extension point
where a user registers their own ``Runner`` subclass. ``OBJECTIVES`` maps an
objective name (``"mpnn"``, ``"diffusion"``) to its loss reducer, and
`ARCHITECTURE_OBJECTIVES` maps each supported architecture id to its default
objective. An architecture with no grounded, in-repo objective is absent from that
table, so the Runner raises rather than fabricating a loss (DESIGN Principle 13).
"""

from __future__ import annotations

import chanfig


class RunnerRegistry(chanfig.ConfigRegistry):
    r"""Resolve the Runner class from ``config.runner`` — a user extension point."""

    key = "runner"


RUNNERS = RunnerRegistry()
OBJECTIVES: chanfig.Registry = chanfig.Registry()

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
