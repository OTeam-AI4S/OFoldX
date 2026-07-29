r"""
Common lifecycle for registered OFoldX models.
"""

from __future__ import annotations

from typing import ClassVar

import chanfig
from torch import nn
from typing_extensions import Self

from .modules.backend.resolve import PUBLIC_BACKENDS, Backend


class Model(nn.Module):
    r"""
    Base class for OFoldX models.

    Args:
        config: Architecture configuration stored with the model.
    """

    config_class: ClassVar[type[chanfig.Config]]
    config: chanfig.Config
    backend: Backend

    def __init__(self, config: chanfig.Config) -> None:
        super().__init__()
        self.config = config
        self.backend = "auto"

    def post_init(self, *, backend: Backend = "auto") -> None:
        r"""
        Initialize model parameters and select a backend.

        Subclasses may override this hook when initialization requires additional
        model-wide work.

        Args:
            backend: Backend selected after parameter initialization.
        """

        self.initialize_parameters()
        self.set_backend(backend)

    def initialize_parameters(self) -> None:
        r"""
        Initialize parameters owned by the model.

        The default implementation leaves module constructors' initialization
        unchanged. Subclasses may override it for architecture-specific rules.
        """

    def set_backend(self, backend: Backend) -> Self:
        r"""
        Select the backend used by dispatched model operators.

        Args:
            backend: Backend name. ``"auto"`` selects the best available implementation,
                while the remaining values express a model-wide provider
                preference. An operator that does not implement that provider
                uses its public default.

        Returns:
            This model.

        Raises:
            ValueError: If ``backend`` is not supported.
        """

        if backend not in PUBLIC_BACKENDS:
            raise ValueError(f"backend must be one of: {', '.join(sorted(PUBLIC_BACKENDS))}")
        self._set_backend(backend)
        self.backend = backend
        return self

    def _set_backend(self, backend: Backend) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "backend"):
                module.backend = backend
