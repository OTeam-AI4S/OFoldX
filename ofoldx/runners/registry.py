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
    "ofoldx.proteinmpnn": "mpnn",
    "ofoldx.ligandmpnn": "mpnn",
    "ofoldx.nampnn": "mpnn",
    "ofoldx.boltz1": "diffusion",
    "ofoldx.boltz2": "diffusion",
    "ofoldx.boltzgen": "diffusion",
    "ofoldx.protenix": "diffusion",
    "ofoldx.odesign": "denoiser",
}
