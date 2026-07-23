r"""Common lifecycle for registered OFoldX models."""

from __future__ import annotations

from typing import ClassVar, Literal

import chanfig
import torch
from torch import nn
from typing_extensions import Self

ModelBackend = Literal["auto", "torch", "oops"]


class Model(nn.Module):
    r"""Base class for top-level models managed by OFoldX.

    Concrete models retain their own constructor and forward signatures. This class owns
    only the state shared by every registered model: ordinary ``nn.Module`` initialization,
    its concrete config, and the requested public operator-backend policy. Models without
    dispatched operators are naturally unaffected by that policy.
    """

    config_class: ClassVar[type[chanfig.Config]]
    config: chanfig.Config
    backend: ModelBackend

    def __init__(self, config: chanfig.Config) -> None:
        super().__init__()
        self.config = config
        self.backend = "auto"

    def post_init(self, *, backend: ModelBackend = "auto") -> None:
        r"""Finish model-specific initialization and apply the initial backend policy."""

        self.initialize_parameters()
        self.set_backend(backend)

    def initialize_parameters(self) -> None:
        r"""Apply additional model-level initialization after child construction.

        PyTorch modules already initialize their own parameters during construction,
        so the default hook intentionally does nothing. Architectures override this
        hook only when they own additional initialization rules.
        """

    @property
    def device(self) -> torch.device:
        r"""Device of this model's parameters or buffers."""

        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""Floating-point dtype used by this model."""

        for tensor in self.parameters():
            if tensor.is_floating_point() or tensor.is_complex():
                return tensor.dtype
        for tensor in self.buffers():
            if tensor.is_floating_point() or tensor.is_complex():
                return tensor.dtype
        return torch.get_default_dtype()

    def set_backend(self, backend: ModelBackend) -> Self:
        r"""Select the model's public PyTorch or OOps operator path."""

        if backend not in ("auto", "torch", "oops"):
            raise ValueError("backend must be exactly one of: auto, torch, oops")
        self._set_backend(backend)
        self.backend = backend
        return self

    def _set_backend(self, backend: ModelBackend) -> None:
        r"""Apply a validated backend to model-specific operator owners."""
