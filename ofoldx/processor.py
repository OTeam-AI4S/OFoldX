# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Common runtime placement for OFoldX Processors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import chanfig
import torch

from .errors import RequestRef, fail

if TYPE_CHECKING:
    from .model import Model


@dataclass(frozen=True)
class PreparationSpec:
    r"""Frozen identity of a processor's persistent representation configuration."""

    processor: str
    configuration: str


class Processor:
    r"""
    Base class for task data preparation, configuration, and tensor placement.

    ``config_class`` owns the Processor's repository configuration schema.
    ``config`` stores one instance of that schema; both are ``None`` for
    configuration-free Processors.

    Args:
        device: Device used for newly created tensors. Defaults to CPU.
        dtype: Floating-point dtype used for newly created tensors. Defaults to
            ``torch.float32``.

    Notes:
        `prepare_input` builds the family's features and preparation context.
        `prepare_target` aligns a concrete truth value against that same context.
        `prepare` composes the two and returns an ordinary `(input, target)` tuple;
        its target is None when no external truth is supplied. These operations
        describe data, not train/eval/inference modes.

        Preparation and execution randomness are separate. The concrete objective
        or sampler constructs teacher history, masking, noise, and other execution
        variables using its own explicit scientific options and generator.
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

    @property
    def preparation_spec(self) -> PreparationSpec:
        r"""Snapshot representation settings independently of tensor placement and random-stream state."""
        values = {} if self.config is None else dict(self.config)
        return PreparationSpec(
            f"{type(self).__module__}.{type(self).__qualname__}",
            json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    def _validate_prepared(self, prepared: Any) -> None:
        if prepared.context.spec != self.preparation_spec:
            fail("value_invalid", RequestRef("prepared", "incompatible processor representation"))

    def prepare_input(self, *args: Any, **options: Any) -> Any:
        r"""Prepare task features and canonical correspondence.

        Concrete families expose their scientific arguments on this operation.
        """
        raise NotImplementedError(f"{type(self).__name__} has no input preparation")

    def prepare_target(self, *args: Any, **options: Any) -> Any:
        r"""Align truth to an existing prepared input without repeating its preparation.

        Families implement the target types supported by their scientific tasks.
        The default implementation raises an error when target preparation is not supported.
        """
        fail("model_capacity", RequestRef("target", f"{type(self).__name__} supervision"))

    def prepare(self, *args: Any, **options: Any) -> tuple[Any, Any | None]:
        r"""Prepare a sample as an ordinary `(input, target)` pair.

        Without an explicit target the second item is None. A self-supervised
        caller can explicitly provide the source as target, or use an objective
        that constructs its own labels from observations such as an unmasked MSA.
        Training and evaluation share this data operation.
        """
        target = options.pop("target", None)
        prepared = self.prepare_input(*args, **options)
        truth = None if target is None else self.prepare_target(target, prepared=prepared)
        return prepared, truth

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

    def validate_model_input_layout(self, model: Model) -> None:
        r"""Validate Processor settings that determine the Model's tensor layout.

        Most Processors have no configurable layout. Families whose Processor
        configuration changes tensor ranks or fixed dimensions override this hook.
        """
