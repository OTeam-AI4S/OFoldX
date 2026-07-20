r"""Training objectives — the loss reducers, keyed by objective name.

The built-in objectives (and any family-specific collate) register themselves
into the registries in :mod:`ofoldx.runners.registry` when their module is
imported; :func:`~ofoldx.runners.registry.build_objective` /
:func:`~ofoldx.runners.registry.build_collate` import them on demand, so this
package stays light enough for ``registry`` to import :class:`Objective` without
a cycle.
"""

from .base import Objective

__all__ = ["Objective"]
