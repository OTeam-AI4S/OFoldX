r"""
Common lifecycle for registered OFoldX models.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import chanfig
import torch
from torch import nn
from typing_extensions import Self

#: Backend request accepted by [`Model.set_backend`][].
ModelBackend = Literal["auto", "eager", "oops"]


class Model(nn.Module):
    r"""
    Base class for OFoldX models.

    Args:
        config: Architecture configuration stored with the model.
    """

    config_class: ClassVar[type[chanfig.Config]]
    config: chanfig.Config
    backend: ModelBackend

    def __init__(self, config: chanfig.Config) -> None:
        super().__init__()
        self.config = config
        self.backend = "auto"

    def post_init(self, *, backend: ModelBackend = "auto") -> None:
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

    @property
    def device(self) -> torch.device:
        r"""
        Return the device of this model's parameters or buffers.

        Returns:
            The first parameter or buffer device, or CPU when the model has neither.
        """

        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Return the floating-point dtype used by this model.

        Returns:
            The first floating-point or complex parameter or buffer dtype, or the
            default PyTorch dtype when the model has neither.
        """

        for tensor in self.parameters():
            if tensor.is_floating_point() or tensor.is_complex():
                return tensor.dtype
        for tensor in self.buffers():
            if tensor.is_floating_point() or tensor.is_complex():
                return tensor.dtype
        return torch.get_default_dtype()

    def set_backend(self, backend: ModelBackend) -> Self:
        r"""
        Select the backend used by dispatched model operators.

        Args:
            backend: Backend name. ``"auto"`` selects the best available implementation,
                while the remaining values request a specific implementation.

        Returns:
            This model.

        Raises:
            ValueError: If ``backend`` is not supported.
        """

        if backend not in ("auto", "eager", "oops"):
            raise ValueError("backend must be exactly one of: auto, eager, oops")
        self._set_backend(backend)
        self.backend = backend
        return self

    def _set_backend(self, backend: ModelBackend) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "backend"):
                module.backend = backend
