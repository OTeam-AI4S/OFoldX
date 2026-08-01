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
Common lifecycle for registered OFoldX models.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NotRequired, Protocol, Self, TypeAlias, cast

import chanfig
from tensordict import TypedTensorDict
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .compilation import CompilationRegionGroups, CompilationRegionOwner
from .modules.backend.resolve import PUBLIC_BACKENDS, Backend

if TYPE_CHECKING:
    OptionalTensor: TypeAlias = Tensor | None
else:
    OptionalTensor = NotRequired[Tensor]


class TensorCollection(TypedTensorDict, dict):
    r"""Typed tensor collection discoverable by PyTorch's ``dict``-based output walkers.

    ``TypedTensorDict`` supplies the storage and mapping methods. The nominal
    ``dict`` base lets DDP find output tensors when determining unused
    parameters; PyTorch's walker does not currently accept arbitrary mappings.
    """

    def __getattr__(self, name: str) -> Any:
        if name in type(self).__optional_keys__:
            return self.get(name)
        return super().__getattr__(name)


if TYPE_CHECKING:
    ShadowTensorCollection: TypeAlias = TensorCollection
else:
    ShadowTensorCollection = TensorCollection["shadow"]


class _GradientCheckpointingModule(Protocol):
    gradient_checkpointing: bool
    _gradient_checkpointing_func: Callable[..., object]


class _BackendModule(Protocol):
    backend: Backend


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

    def compilation_region_groups(self) -> CompilationRegionGroups:
        r"""Collect and validate regions declared explicitly by module owners."""

        root_modules = tuple(self.modules())
        root_module_ids = {id(module) for module in root_modules}
        groups = tuple(
            group
            for module in root_modules
            if isinstance(module, CompilationRegionOwner)
            for group in module.local_compilation_region_groups()
        )

        seen: set[int] = set()
        regions: list[nn.Module] = []
        for group in groups:
            if not group:
                raise ValueError("compilation region groups must be non-empty")
            for region in group:
                if not isinstance(region, nn.Module) or id(region) not in root_module_ids:
                    raise ValueError("compilation regions must belong to the model")
                if id(region) in seen:
                    raise ValueError("a compilation region may appear only once")
                seen.add(id(region))
                regions.append(region)

        region_ids = {id(region) for region in regions}
        for region in regions:
            descendant_ids = {id(descendant) for descendant in tuple(region.modules())[1:]}
            if descendant_ids & region_ids:
                raise ValueError("compilation regions may not contain one another")
        return groups

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
                cast(_BackendModule, module).backend = backend

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
