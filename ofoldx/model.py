r"""
Common lifecycle for registered OFoldX models.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar, Protocol, cast

import chanfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import Self

from .modules.backend.resolve import PUBLIC_BACKENDS, Backend


class _GradientCheckpointingModule(Protocol):
    gradient_checkpointing: bool
    _gradient_checkpointing_func: Callable[..., object]


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

    def post_init(self, *, backend: Backend | None = None) -> None:
        r"""
        Initialize model parameters and select a backend.

        Subclasses may override this hook when initialization requires additional
        model-wide work.

        Args:
            backend: Optional model-wide Backend selected after parameter
                initialization. ``None`` preserves each child Module's
                construction-time backend.
        """

        self.initialize_parameters()
        if backend is not None:
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

    @property
    def is_gradient_checkpointing(self) -> bool:
        r"""
        Whether gradient checkpointing is enabled on any repeated module.
        """

        return any(
            hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing for module in self.modules()
        )

    def gradient_checkpointing_enable(self) -> Self:
        r"""
        Enable gradient checkpointing on supported repeated modules.

        Checkpointing remains inactive in evaluation and no-grad execution.

        Returns:
            This model.

        Raises:
            ValueError: If this model has no checkpointable repeated module.
        """

        checkpoint_function = partial(checkpoint, use_reentrant=False)
        found = False
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                checkpointing_module = cast(_GradientCheckpointingModule, module)
                checkpointing_module._gradient_checkpointing_func = checkpoint_function
                checkpointing_module.gradient_checkpointing = True
                found = True
        if not found:
            raise ValueError(f"{type(self).__name__} does not support gradient checkpointing")
        return self

    def gradient_checkpointing_disable(self) -> Self:
        r"""
        Disable gradient checkpointing on every repeated module.

        Returns:
            This model.
        """

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                cast(_GradientCheckpointingModule, module).gradient_checkpointing = False
        return self
