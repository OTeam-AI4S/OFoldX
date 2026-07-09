r"""
Typed representation carriers with semantic axes.

This domain's dominant bug is an axis/shape mismatch -- ``N_atom`` used where ``N_token`` is meant, a
``pair_i`` axis swapped with ``pair_j``, a rotation matrix passed as a translation -- not a class-name slip.
These carriers make the **boundary** typed: each wraps a raw tensor plus its semantic axis contract and
checks it once at construction (under ``__debug__`` only, so production hot paths pay nothing). They are
typed *data*; interior safety comes from typed *operations* -- the named-axis methods here
([`Coords.centered`][ofoldx.modules.representation.Coords.centered],
[`Frames.apply`][ofoldx.modules.representation.Frames.apply]) -- so callers never hand-roll a ``.transpose``
on a coordinate or frame tensor. Carriers hold raw tensors, so ``.tensor`` / ``.rotations`` drop straight
back into eager PyTorch and ``torch.compile``; nothing is wrapped element-wise.

Examples:
    >>> import torch
    >>> coords = Coords(torch.zeros(4, 3))
    >>> coords.num_points
    4
    >>> Coords(torch.zeros(4, 2))
    Traceback (most recent call last):
        ...
    ValueError: Coords last axis must be 3 (xyz), got (4, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import register_pytree_node


def _require(condition: bool, message: str) -> None:
    r"""Raise ``ValueError`` when ``condition`` is false, but only in debug builds (``python -O`` skips it)."""

    if __debug__ and not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Coords:
    r"""
    Cartesian coordinates ``[..., point, 3]`` -- the trailing ``coord`` axis is xyz.

    Attributes:
        tensor: The underlying ``[..., point, 3]`` coordinate tensor.

    Examples:
        >>> import torch
        >>> bool(Coords(torch.arange(6.0).reshape(2, 3)).centered().tensor.mean(dim=-2).abs().max() < 1e-6)
        True
    """

    tensor: Tensor

    def __post_init__(self) -> None:
        _require(self.tensor.shape[-1] == 3, f"Coords last axis must be 3 (xyz), got {tuple(self.tensor.shape)}")

    @property
    def num_points(self) -> int:
        r"""Number of points on the ``point`` axis (the ``-2`` axis)."""

        return self.tensor.shape[-2]

    def centered(self) -> Coords:
        r"""Return coords translated so the ``point`` axis has zero mean."""

        return Coords(self.tensor - self.tensor.mean(dim=-2, keepdim=True))

    def to(self, *args: Any, **kwargs: Any) -> Coords:
        r"""Move/cast the underlying tensor (``Tensor.to`` signature)."""

        return Coords(self.tensor.to(*args, **kwargs))


@dataclass(frozen=True, slots=True)
class Frames:
    r"""
    Rigid frames: rotations ``[..., 3, 3]`` and translations ``[..., 3]`` sharing leading axes.

    Attributes:
        rotations: ``[..., 3, 3]`` rotation matrices.
        translations: ``[..., 3]`` translation vectors.

    Examples:
        >>> import torch
        >>> frames = Frames.identity(2)
        >>> local = Coords(torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]))
        >>> torch.allclose(frames.apply(local).tensor, local.tensor)
        True
    """

    rotations: Tensor
    translations: Tensor

    def __post_init__(self) -> None:
        _require(
            self.rotations.shape[-2:] == (3, 3),
            f"Frames rotations must end with (3, 3), got {tuple(self.rotations.shape)}",
        )
        _require(
            self.translations.shape[-1] == 3,
            f"Frames translations must end with 3, got {tuple(self.translations.shape)}",
        )
        _require(
            self.rotations.shape[:-2] == self.translations.shape[:-1],
            "Frames rotations and translations must share leading (batch) axes: "
            f"{tuple(self.rotations.shape[:-2])} vs {tuple(self.translations.shape[:-1])}",
        )

    @classmethod
    def identity(cls, *batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Frames:
        r"""Return identity frames with the given leading ``batch`` axes."""

        rotations = torch.eye(3, device=device, dtype=dtype).expand(*batch, 3, 3)
        translations = torch.zeros(*batch, 3, device=device, dtype=dtype)
        return cls(rotations, translations)

    def apply(self, coords: Coords) -> Coords:
        r"""
        Map local ``coords`` into the global frame: ``R @ x + t`` (rotation over the ``coord`` axis).

        Args:
            coords: Local coordinates ``[..., point, 3]`` sharing this frame's leading axes.

        Returns:
            Global coordinates ``[..., point, 3]``.
        """

        rotated = torch.einsum("...ij,...pj->...pi", self.rotations, coords.tensor)
        return Coords(rotated + self.translations.unsqueeze(-2))

    def compose(self, other: Frames) -> Frames:
        r"""Return ``self ∘ other``: apply ``other`` then ``self``."""

        rotations = self.rotations @ other.rotations
        translations = torch.einsum("...ij,...j->...i", self.rotations, other.translations) + self.translations
        return Frames(rotations, translations)

    def inverse(self) -> Frames:
        r"""Return the inverse rigid transform ``(Rᵀ, -Rᵀ t)``."""

        inverse_rotations = self.rotations.transpose(-1, -2)
        inverse_translations = -torch.einsum("...ij,...j->...i", inverse_rotations, self.translations)
        return Frames(inverse_rotations, inverse_translations)

    def to(self, *args: Any, **kwargs: Any) -> Frames:
        r"""Move/cast the underlying tensors (``Tensor.to`` signature)."""

        return Frames(self.rotations.to(*args, **kwargs), self.translations.to(*args, **kwargs))


@dataclass(frozen=True, slots=True)
class Single:
    r"""
    Single (token) representation ``[..., token, channel]``.

    Attributes:
        tensor: The underlying ``[..., token, channel]`` tensor.

    Examples:
        >>> import torch
        >>> Single(torch.zeros(1, 8, 32)).num_tokens
        8
    """

    tensor: Tensor

    def __post_init__(self) -> None:
        _require(self.tensor.ndim >= 2, f"Single needs a (token, channel) tail, got {tuple(self.tensor.shape)}")

    @property
    def num_tokens(self) -> int:
        r"""Length of the ``token`` axis (the ``-2`` axis)."""

        return self.tensor.shape[-2]

    @property
    def channels(self) -> int:
        r"""Width of the ``channel`` axis (the ``-1`` axis)."""

        return self.tensor.shape[-1]

    def to(self, *args: Any, **kwargs: Any) -> Single:
        r"""Move/cast the underlying tensor (``Tensor.to`` signature)."""

        return Single(self.tensor.to(*args, **kwargs))


@dataclass(frozen=True, slots=True)
class Pair:
    r"""
    Pair representation ``[..., token_i, token_j, channel]`` with equal ``i``/``j`` axes.

    Attributes:
        tensor: The underlying ``[..., token_i, token_j, channel]`` tensor.

    Examples:
        >>> import torch
        >>> pair = Pair(torch.randn(1, 4, 4, 8))
        >>> pair.transpose_ij().tensor.shape
        torch.Size([1, 4, 4, 8])
        >>> bool((pair.symmetrize().tensor == pair.symmetrize().transpose_ij().tensor).all())
        True
    """

    tensor: Tensor

    def __post_init__(self) -> None:
        _require(self.tensor.ndim >= 3, f"Pair needs an (i, j, channel) tail, got {tuple(self.tensor.shape)}")
        _require(
            self.tensor.shape[-3] == self.tensor.shape[-2],
            f"Pair i and j axes must have equal length, got {tuple(self.tensor.shape)}",
        )

    @property
    def num_tokens(self) -> int:
        r"""Length of the (equal) ``token_i``/``token_j`` axes."""

        return self.tensor.shape[-2]

    def transpose_ij(self) -> Pair:
        r"""Swap the ``token_i`` and ``token_j`` axes -- the named alternative to a hand-rolled transpose."""

        return Pair(self.tensor.transpose(-3, -2))

    def symmetrize(self) -> Pair:
        r"""Return the ``i``/``j``-symmetric average ``(P + Pᵀ) / 2``."""

        return Pair(0.5 * (self.tensor + self.tensor.transpose(-3, -2)))

    def to(self, *args: Any, **kwargs: Any) -> Pair:
        r"""Move/cast the underlying tensor (``Tensor.to`` signature)."""

        return Pair(self.tensor.to(*args, **kwargs))


def _register_carrier(cls: type, fields: tuple[str, ...]) -> None:
    r"""
    Register a frozen tensor carrier as a ``torch.utils._pytree`` node.

    Flattening yields the carrier's tensor fields as pytree *children* (so ``tree_map`` and
    ``torch.compile`` see straight through to the raw tensors with no graph break) and the field-name
    tuple as the static *context* (the axis-meta identity used to rebuild the carrier). Unflattening
    reconstructs ``cls(**{name: child})``, re-running the carrier's ``__post_init__`` axis checks.

    Args:
        cls: The frozen carrier dataclass to register.
        fields: The carrier's tensor field names, in construction order.
    """

    def _flatten(carrier: Any) -> tuple[list[Tensor], tuple[str, ...]]:
        return [getattr(carrier, name) for name in fields], fields

    def _unflatten(children: Any, context: tuple[str, ...]) -> Any:
        return cls(**dict(zip(context, children)))

    register_pytree_node(cls, _flatten, _unflatten)


_register_carrier(Coords, ("tensor",))
_register_carrier(Frames, ("rotations", "translations"))
_register_carrier(Single, ("tensor",))
_register_carrier(Pair, ("tensor",))
