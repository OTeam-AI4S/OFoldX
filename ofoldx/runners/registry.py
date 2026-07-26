r"""
Registries for runners and training objectives.
"""

from __future__ import annotations

import chanfig


class RunnerRegistry(chanfig.ConfigRegistry):
    r"""
    Resolve a runner class from ``config.runner``.
    """

    key = "runner"


#: Runner implementations keyed by their configuration name.
RUNNERS = RunnerRegistry()
#: Training objectives keyed by objective name.
OBJECTIVES: chanfig.Registry = chanfig.Registry()

# Architecture ID to default objective name.
ARCHITECTURE_OBJECTIVES: dict[str, str] = {
    "proteinmpnn": "mpnn",
    "ligandmpnn": "mpnn",
    "nampnn": "mpnn",
    "boltz1": "diffusion",
    "boltz2": "diffusion",
    "boltzgen": "diffusion",
    "protenix": "diffusion",
    "odesign": "denoiser",
}
