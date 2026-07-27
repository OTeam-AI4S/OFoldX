r"""
Registry for runner implementations.
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
