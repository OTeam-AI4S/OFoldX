r"""
Common runtime placement for OFoldX Processors.
"""

from __future__ import annotations

from typing import ClassVar

import chanfig
import torch
from typing_extensions import Self


class UnresolvedTrainingCropError(ValueError):
    r"""Raised when a training crop contains no resolved model-visible atoms."""


class Processor:
    r"""
    Base class for processors with a default tensor device and dtype.

    ``config_class`` owns the Processor's repository configuration schema.
    ``config`` stores one instance of that schema; both are ``None`` for
    configuration-free Processors.

    Args:
        device: Device used for newly created tensors. Defaults to CPU.
        dtype: Floating-point dtype used for newly created tensors. Defaults to
            ``torch.float32``.
    """

    config_class: ClassVar[type[chanfig.Config] | None] = None
    config: chanfig.Config | None = None
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Update the tensor device and dtype used by this processor.

        Args:
            device: New tensor device. ``None`` preserves the current device.
            dtype: New floating-point dtype. ``None`` preserves the current dtype.

        Returns:
            This processor after in-place placement update.
        """

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self
