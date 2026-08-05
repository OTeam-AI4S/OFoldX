r"""
Exponential moving-average shadow models for training runners.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from torch import nn
from typing_extensions import Self

from ..modules.backend.resolve import Backend


def _unwrap_model(model: nn.Module) -> nn.Module:
    r"""Remove supported execution wrappers without relying on their state-dict key prefixes."""

    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
            model = model.module
            continue
        original = getattr(model, "_orig_mod", None)
        if isinstance(original, nn.Module):
            model = original
            continue
        break
    return model


class EMAShadowModel(nn.Module):
    r"""
    A frozen, forwardable exponential moving average of a model.

    The wrapper intentionally exposes the wrapped model's unprefixed state dict
    so EMA checkpoints have the same schema as ordinary model checkpoints.

    Args:
        model: Live model whose current parameters and buffers initialize the shadow.
            DDP, DataParallel, and ``torch.compile`` wrappers are normalized first.
        decay: Fixed EMA decay in the open interval ``(0, 1)``.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__()
        if isinstance(decay, bool) or not isinstance(decay, Real):
            raise TypeError("EMA decay must be a finite real number strictly between 0 and 1")
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be finite and strictly between 0 and 1, got {decay!r}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"EMA source must be an nn.Module, got {type(model).__name__}")

        self.decay = decay
        self.model = copy.deepcopy(_unwrap_model(model))
        self.model.requires_grad_(False)

    @property
    def config(self) -> Any:
        r"""Expose the underlying model configuration without a second copy."""

        return getattr(self.model, "config")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        r"""Delegate the registered model family's inference entry point."""

        sample = getattr(self.model, "sample", None)
        if not callable(sample):
            raise AttributeError(f"{type(self.model).__name__} does not provide a callable `sample` method")
        return sample(*args, **kwargs)

    def set_backend(self, backend: Backend) -> Self:
        r"""Select the same operator backend on the shadow model."""

        set_backend = getattr(self.model, "set_backend", None)
        if not callable(set_backend):
            raise AttributeError(f"{type(self.model).__name__} does not provide a callable `set_backend` method")
        set_backend(backend)
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        r"""Return the model state directly, without an EMA-wrapper key prefix."""

        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        r"""Load an ordinary model state directly into the shadow."""

        result = self.model.load_state_dict(state_dict, *args, **kwargs)
        self.model.requires_grad_(False)
        return result

    @staticmethod
    def _validate_state(
        target: nn.Module,
        source: nn.Module,
    ) -> tuple[
        dict[str, nn.Parameter],
        dict[str, nn.Parameter],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        target_parameters = dict(target.named_parameters())
        source_parameters = dict(source.named_parameters())
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())

        EMAShadowModel._validate_names("parameter", target_parameters, source_parameters)
        EMAShadowModel._validate_names("buffer", target_buffers, source_buffers)
        for kind, target_values, source_values in (
            ("parameter", target_parameters, source_parameters),
            ("buffer", target_buffers, source_buffers),
        ):
            for name, target_value in target_values.items():
                source_value = source_values[name]
                if target_value.shape != source_value.shape:
                    raise ValueError(
                        f"EMA {kind} {name!r} shape mismatch: shadow={tuple(target_value.shape)}, "
                        f"source={tuple(source_value.shape)}"
                    )
                if target_value.dtype != source_value.dtype:
                    raise ValueError(
                        f"EMA {kind} {name!r} dtype mismatch: shadow={target_value.dtype}, "
                        f"source={source_value.dtype}"
                    )
                if target_value.device != source_value.device:
                    raise ValueError(
                        f"EMA {kind} {name!r} device mismatch: shadow={target_value.device}, "
                        f"source={source_value.device}"
                    )
        return target_parameters, source_parameters, target_buffers, source_buffers

    @staticmethod
    def _validate_names(kind: str, target: Mapping[str, Any], source: Mapping[str, Any]) -> None:
        target_names = set(target)
        source_names = set(source)
        if target_names == source_names:
            return
        missing = sorted(target_names - source_names)
        unexpected = sorted(source_names - target_names)
        raise ValueError(
            f"EMA {kind} names do not match: missing from source={missing}, unexpected in source={unexpected}"
        )

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        r"""Apply one fixed-decay EMA update and copy all buffers exactly."""

        if not isinstance(source, nn.Module):
            raise TypeError(f"EMA source must be an nn.Module, got {type(source).__name__}")
        source = _unwrap_model(source)
        target_parameters, source_parameters, target_buffers, source_buffers = self._validate_state(
            self.model, source
        )
        one_minus_decay = 1.0 - self.decay
        for name, target in target_parameters.items():
            current = source_parameters[name]
            if torch.is_floating_point(target) or torch.is_complex(target):
                target.mul_(self.decay).add_(current, alpha=one_minus_decay)
            else:
                target.copy_(current)
        for name, target in target_buffers.items():
            target.copy_(source_buffers[name])

    @torch.no_grad()
    def copy_from(self, source: nn.Module) -> None:
        r"""Synchronize every parameter and buffer exactly from a live model."""

        if not isinstance(source, nn.Module):
            raise TypeError(f"EMA source must be an nn.Module, got {type(source).__name__}")
        source = _unwrap_model(source)
        target_parameters, source_parameters, target_buffers, source_buffers = self._validate_state(
            self.model, source
        )
        for name, target in target_parameters.items():
            target.copy_(source_parameters[name])
        for name, target in target_buffers.items():
            target.copy_(source_buffers[name])
