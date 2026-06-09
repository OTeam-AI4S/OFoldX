r"""
Runner registry.
"""

from __future__ import annotations

from chanfig import ConfigRegistry


class Registry(ConfigRegistry):
    r"""
    Config-driven registry that maps a ``config.runner`` string to its runner class.

    A [`ConfigRegistry`][chanfig.ConfigRegistry] specialized on the ``runner`` key, so a config carrying
    ``runner="ofoldx"`` resolves to the class registered under that name. The module-level
    [`RUNNERS`][ofoldx.runners.registry.RUNNERS] singleton is the shared instance runner classes register on.

    Attributes:
        key: Config field inspected to look up the registered class (``"runner"``).

    Examples:
        >>> Registry.key
        'runner'
    """

    key = "runner"


RUNNERS = Registry()
