r"""Compatibility helpers for tensorclass-backed domain values."""

from __future__ import annotations

from typing import TypeVar

from tensordict import tensorclass
from typing_extensions import dataclass_transform


_T = TypeVar("_T")


def _setstate_frozen(self: object, state: dict[str, object]) -> None:
    r"""Restore tensorclass internals without invoking frozen dataclass guards."""

    object.__setattr__(self, "_tensordict", state.get("tensordict"))
    object.__setattr__(self, "_non_tensordict", state.get("non_tensordict"))


@dataclass_transform(frozen_default=True)
def frozen_tensorclass(cls: type[_T]) -> type[_T]:
    r"""Create a frozen tensorclass whose generated pickle hook can restore state."""

    decorated = tensorclass(cls, frozen=True)
    # TensorDict 0.13.0's generated hook assigns these attributes normally,
    # which is rejected by the frozen dataclass ``__setattr__``.
    setattr(decorated, "__setstate__", _setstate_frozen)
    return decorated
