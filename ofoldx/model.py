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
    its concrete config, and selection of its public operator backend.
    """

    config_class: ClassVar[type[chanfig.Config]]
    config: chanfig.Config
    backend: ModelBackend

    def __init__(self, config: chanfig.Config) -> None:
        super().__init__()
        self.config = config
        self.backend = "auto"

    @torch.no_grad()
    def reset_parameters(self) -> None:
        r"""Reset child modules with their native PyTorch initialization."""

        def reset(module: nn.Module) -> None:
            reset_parameters = getattr(module, "reset_parameters", None)
            if reset_parameters is not None:
                reset_parameters()

        for module in self.children():
            module.apply(reset)

    def set_backend(self, backend: ModelBackend) -> Self:
        r"""Select the model's public PyTorch or OOps operator path."""

        if backend not in ("auto", "torch", "oops"):
            raise ValueError("backend must be exactly one of: auto, torch, oops")
        self._set_backend(backend)
        self.backend = backend
        return self

    def _set_backend(self, backend: ModelBackend) -> None:
        r"""Apply a validated backend to model-specific operator owners."""

        pass
